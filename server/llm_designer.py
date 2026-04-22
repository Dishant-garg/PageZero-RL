import json
import os
import random

from dotenv import load_dotenv

from .config import (
    POSTGRES_CONTAINER, REDIS_CONTAINER, APP_CONTAINER,
    HARD_SCENARIO_THRESHOLD,
)

load_dotenv()

WARMUP_SCENARIOS = [
    {
        "name": "runaway-query",
        "difficulty": 0.1,
        "layer": "database",
        "alert": "CRITICAL: API p99 latency > 5s, PostgreSQL CPU at 95%",
        "inject_commands": [
            f'docker exec {POSTGRES_CONTAINER} psql -U sre -d production -c "SELECT pg_sleep(300), * FROM orders o1 CROSS JOIN orders o2 LIMIT 1;" &'
        ],
        "root_cause": "Runaway CROSS JOIN query consuming all CPU",
        "expected_fix": ["pg_cancel_query"],
    },
    {
        "name": "missing-index-load",
        "difficulty": 0.2,
        "layer": "database",
        "alert": "WARNING: /api/orders endpoint p99 > 3s",
        "inject_commands": [
            "for i in $(seq 1 20); do curl -s http://localhost:5000/api/orders/user_${i}@company.com > /dev/null & done"
        ],
        "root_cause": "Missing index on orders.user_email causing sequential scans under load",
        "expected_fix": ["pg_create_index"],
    },
    {
        "name": "redis-oom",
        "difficulty": 0.2,
        "layer": "cache",
        "alert": "CRITICAL: Redis memory usage > 95%, OOM errors in app logs",
        "inject_commands": [
            f"docker exec {REDIS_CONTAINER} redis-cli DEBUG SET-ACTIVE-EXPIRE 0",
            # Pre-generate data once instead of 1000 sequential docker exec calls
            f"docker exec {REDIS_CONTAINER} sh -c 'for i in $(seq 1 1000); do redis-cli SET garbage_$i $(head -c 500 /dev/urandom | base64 | head -c 500); done'",
        ],
        "root_cause": "Redis memory exhausted by orphaned keys",
        "expected_fix": ["redis_flush_db"],
    },
    {
        "name": "redis-cache-drop",
        "difficulty": 0.3,
        "layer": "cache",
        "alert": "WARNING: Cache miss rate spikes to 100%",
        "inject_commands": [
            f"docker exec {REDIS_CONTAINER} redis-cli FLUSHALL"
        ],
        "root_cause": "Cache was accidentally dropped, rebuilding data slowly",
        "expected_fix": ["curl_endpoint"],
    },
    {
        "name": "connection-pool-exhausted",
        "difficulty": 0.4,
        "layer": "database",
        "alert": "CRITICAL: 'too many connections' errors, new requests timing out",
        "inject_commands": [
            f"for i in $(seq 1 50); do docker exec {POSTGRES_CONTAINER} psql -U sre -d production -c 'SELECT pg_sleep(600)' & done"
        ],
        "root_cause": "Leaked connections exhausting PostgreSQL max_connections",
        "expected_fix": ["pg_cancel_query"],
    },
]

# Sort warmup scenarios by difficulty for ordered selection
WARMUP_SCENARIOS.sort(key=lambda s: s["difficulty"])


HARD_SCENARIOS = [
    {
        "name": "cascading-lock-timeout",
        "difficulty": 0.7,
        "layer": "cross_layer",
        "alert": "CRITICAL: API Latency > 10s, entire application unresponsive.",
        "inject_commands": [
            f'docker exec {POSTGRES_CONTAINER} psql -U sre -d production -c "BEGIN; LOCK TABLE orders IN EXCLUSIVE MODE; SELECT pg_sleep(300);" &',
            f"docker exec {REDIS_CONTAINER} redis-cli FLUSHDB",
        ],
        "root_cause": "Exclusive lock on orders table created a connection pool queue, and cache was empty.",
        "expected_fix": ["pg_cancel_query"],
    },
    {
        "name": "full-stack-meltdown",
        "difficulty": 0.9,
        "layer": "cross_layer",
        "alert": "CRITICAL: Multiple services degraded — app 503s, DB locks, Redis OOM.",
        "inject_commands": [
            # Lock the DB
            f'docker exec {POSTGRES_CONTAINER} psql -U sre -d production -c "BEGIN; LOCK TABLE orders IN EXCLUSIVE MODE; SELECT pg_sleep(300);" &',
            # Fill Redis
            f"docker exec {REDIS_CONTAINER} sh -c 'for i in $(seq 1 800); do redis-cli SET meltdown_$i $(head -c 500 /dev/urandom | base64 | head -c 500); done'",
        ],
        "root_cause": "Simultaneous DB lock and Redis memory exhaustion causing cascading failures across all layers.",
        "expected_fix": ["pg_cancel_query", "redis_flush_db"],
    },
]


class LLMDesigner:
    """Uses Gemini API to generate complex scenarios, falling back to
    difficulty-ordered programmatic scenarios if unavailable."""

    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if self.api_key and "your_" not in self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
            except ImportError:
                self.client = None
        else:
            self.client = None

    def design(self, skill_profile: dict, difficulty: float,
               *, use_warmup: bool = True, weakest_layer: str = "") -> dict:
        """Generate a scenario matched to the current difficulty.

        Args:
            skill_profile: per-layer skill scores
            difficulty: current difficulty level (0.0–1.0)
            use_warmup: if True, pick from warmup pool; else try LLM/hard pool
            weakest_layer: optional hint to bias toward agent's weakest layer
        """
        fallback = self._get_fallback(difficulty, weakest_layer)

        if not self.client or use_warmup:
            return fallback

        # Try LLM-generated scenario for non-warmup
        prompt = f"""You are generating an SRE incident scenario for an RL agent.
The architecture is: Postgres ({POSTGRES_CONTAINER}), Redis ({REDIS_CONTAINER}), Flask App ({APP_CONTAINER}).
Agent's skill profile: {json.dumps(skill_profile)}
Target difficulty: {difficulty} (0.0 to 1.0)
Agent's weakest layer: {weakest_layer or "unknown"}

Generate a scenario in strict JSON format matching exactly this schema:
{{
    "name": "string (kebab-case)",
    "difficulty": {difficulty},
    "layer": "database|cache|application|cross_layer",
    "alert": "string (e.g. CRITICAL: ...)",
    "inject_commands": ["list of bash commands to execute on the docker container to cause the incident"],
    "root_cause": "string explaining the issue",
    "expected_fix": ["list of tool names from (pg_cancel_query, pg_create_index, pg_vacuum, redis_flush_db, docker_restart, rollback_deploy)"]
}}

Ensure the `inject_commands` are valid shell commands using `docker exec` against the containers.
IMPORTANT: Return ONLY the raw JSON string without any markdown formatting like ```json.
"""
        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            raw_json = response.text.replace("```json", "").replace("```", "").strip()
            scenario = json.loads(raw_json)
            for key in ["name", "difficulty", "layer", "alert", "inject_commands"]:
                if key not in scenario:
                    return fallback
            # Ensure expected_fix exists
            if "expected_fix" not in scenario:
                scenario["expected_fix"] = []
            return scenario
        except Exception as e:
            print(f"Gemini generation failed, using fallback: {e}")
            return fallback

    def _get_fallback(self, difficulty: float, weakest_layer: str = "") -> dict:
        """Select a scenario matched to the current difficulty level.

        - At low difficulty: pick only easy scenarios
        - As difficulty grows: unlock harder scenarios from the pool
        - Above HARD_SCENARIO_THRESHOLD: use hard / cross-layer scenarios
        """
        if difficulty > HARD_SCENARIO_THRESHOLD:
            # Pick from hard scenarios, optionally biased by weakest layer
            candidates = HARD_SCENARIOS
            layer_match = [s for s in candidates if s["layer"] == weakest_layer]
            if layer_match:
                return random.choice(layer_match)
            return random.choice(candidates)

        # Filter warmup scenarios to only those whose difficulty <= current + 0.15
        # This ensures the agent starts with easy tasks and gradually unlocks harder ones
        eligible = [s for s in WARMUP_SCENARIOS if s["difficulty"] <= difficulty + 0.15]
        if not eligible:
            eligible = [WARMUP_SCENARIOS[0]]  # always have at least the easiest

        # Bias toward weakest layer if available
        if weakest_layer:
            layer_match = [s for s in eligible if s["layer"] == weakest_layer]
            if layer_match:
                # 60% chance to pick from weak layer, 40% random
                if random.random() < 0.6:
                    return random.choice(layer_match)

        return random.choice(eligible)
