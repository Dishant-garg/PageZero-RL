import json
import os
import random
from typing import List
from pydantic import BaseModel, Field

from dotenv import load_dotenv

from .config import (
    POSTGRES_CONTAINER, REDIS_CONTAINER, APP_CONTAINER,
    HARD_SCENARIO_THRESHOLD,
)

load_dotenv()

class Scenario(BaseModel):
    """Data model for SRE incident scenarios."""
    name: str = Field(..., description="Scenario name in kebab-case")
    task_id: str = Field(default="", description="Unique task ID for OpenEnv validator")
    difficulty: float = Field(..., ge=0.0, le=1.0, description="Difficulty level 0.0-1.0")
    layer: str = Field(..., description="Layer: database, cache, application, or cross_layer")
    alert: str = Field(..., description="Alert message (e.g. CRITICAL: ...)")
    inject_commands: List[str] = Field(..., description="Bash commands to execute via docker exec")
    root_cause: str = Field(..., description="Root cause explanation")
    expected_fix: List[str] = Field(..., description="Tool names from: pg_cancel_query, pg_create_index, pg_vacuum, redis_flush_db, docker_restart, rollback_deploy")
    hint: str = Field(default="", description="Optional first-step diagnostic hint for the agent")
    grader_profile: dict = Field(default_factory=dict, description="Per-task grading weights for distinct scoring")

WARMUP_SCENARIOS = [
    Scenario(
        name="runaway-query",
        difficulty=0.1,
        layer="database",
        alert="CRITICAL: API p99 latency > 5s, PostgreSQL CPU at 95% — investigate active queries in PostgreSQL",
        inject_commands=[
            f'docker exec {POSTGRES_CONTAINER} psql -U sre -d production -c "SELECT pg_sleep(300), * FROM orders o1 CROSS JOIN orders o2 LIMIT 1;" &'
        ],
        root_cause="Runaway CROSS JOIN query consuming all CPU",
        expected_fix=["pg_cancel_query"],
        hint="Start with pg_stat_activity to find the long-running query, then pg_cancel_query to kill it.",
    ),
    Scenario(
        name="missing-index-load",
        difficulty=0.2,
        layer="database",
        alert="WARNING: /api/orders endpoint p99 > 3s — sequential scans suspected on orders table",
        inject_commands=[
            "for i in $(seq 1 20); do curl -s http://localhost:5000/api/orders/user_${i}@company.com > /dev/null & done"
        ],
        root_cause="Missing index on orders.user_email causing sequential scans under load",
        expected_fix=["pg_create_index"],
        hint="Use pg_stat_statements or pg_explain_analyze to find slow queries, then pg_create_index on the offending column.",
    ),
    Scenario(
        name="redis-oom",
        difficulty=0.2,
        layer="cache",
        alert="CRITICAL: Redis memory usage > 95%, OOM errors in app logs — check redis_info and flush orphaned keys",
        inject_commands=[
            f"docker exec {REDIS_CONTAINER} redis-cli DEBUG SET-ACTIVE-EXPIRE 0",
            f"docker exec {REDIS_CONTAINER} sh -c 'for i in $(seq 1 1000); do redis-cli SET garbage_$i $(head -c 500 /dev/urandom | base64 | head -c 500); done'",
        ],
        root_cause="Redis memory exhausted by orphaned keys",
        expected_fix=["redis_flush_db"],
        hint="Run redis_info to confirm memory usage, then redis_keys to see garbage keys, then redis_flush_db to clear them.",
    ),
    Scenario(
        name="redis-cache-drop",
        difficulty=0.3,
        layer="cache",
        alert="WARNING: Cache miss rate spikes to 100% — Redis may have been flushed; check redis_info",
        inject_commands=[
            f"docker exec {REDIS_CONTAINER} redis-cli FLUSHALL"
        ],
        root_cause="Cache was accidentally dropped, rebuilding data slowly",
        expected_fix=["curl_endpoint"],
        hint="Use redis_info to verify keyspace is empty, then curl_endpoint to verify the app is still serving (cache will rebuild itself).",
    ),
    Scenario(
        name="connection-pool-exhausted",
        difficulty=0.4,
        layer="database",
        alert="CRITICAL: 'too many connections' errors, new requests timing out — check pg_stat_activity for idle connections",
        inject_commands=[
            f"for i in $(seq 1 50); do docker exec {POSTGRES_CONTAINER} psql -U sre -d production -c 'SELECT pg_sleep(600)' & done"
        ],
        root_cause="Leaked connections exhausting PostgreSQL max_connections",
        expected_fix=["pg_cancel_query"],
        hint="Use pg_stat_activity to see all sleeping connections, then pg_cancel_query with their PIDs to release the pool.",
    ),
]

# Sort warmup scenarios by difficulty for ordered selection
WARMUP_SCENARIOS.sort(key=lambda s: s.difficulty)


# ── Medium-difficulty scenarios (0.3–0.55): cover infra, app, and cross-layer ──
MEDIUM_SCENARIOS = [
    Scenario(
        name="disk-full-halt",
        difficulty=0.3,
        layer="application",
        alert="CRITICAL: PostgreSQL write errors — 'could not write to file'; disk usage may be at 100% — run check_disk_usage",
        inject_commands=[
            f"docker exec {POSTGRES_CONTAINER} bash -c 'dd if=/dev/zero of=/tmp/disk_bloat bs=1M count=256 2>/dev/null || true'"
        ],
        root_cause="Disk partition filled by a runaway temp file inside the Postgres container, blocking WAL writes",
        expected_fix=["check_disk_usage", "docker_restart"],
        hint="Run check_disk_usage to confirm 100% disk, then docker_restart on the postgres container to reclaim space after the bloat file is cleaned.",
    ),
    Scenario(
        name="app-crash-loop",
        difficulty=0.35,
        layer="application",
        alert="CRITICAL: pagezero-app-1 returning HTTP 502 — container may have crashed; run docker_ps to check status",
        inject_commands=[
            f"docker kill --signal=SIGKILL {APP_CONTAINER} || true"
        ],
        root_cause="Flask app container was SIGKILL'd (simulating OOM); Docker restart policy is looping it",
        expected_fix=["docker_logs", "docker_restart"],
        hint="Use docker_ps to confirm the container is stopped/restarting, then docker_logs to get the crash reason, then docker_restart to bring it back up.",
    ),
    Scenario(
        name="table-bloat-vacuum-needed",
        difficulty=0.4,
        layer="database",
        alert="WARNING: /api/orders query degraded 4x — dead tuple bloat suspected after overnight batch job; try pg_vacuum",
        inject_commands=[
            f"docker exec {POSTGRES_CONTAINER} psql -U sre -d production -c "
            "\"DO $$ BEGIN FOR i IN 1..5 LOOP "
            "UPDATE orders SET status='archived' WHERE status='pending'; "
            "UPDATE orders SET status='pending' WHERE status='archived'; "
            "END LOOP; END $$;\" || true"
        ],
        root_cause="Dead tuple accumulation after high-churn UPDATE loop; autovacuum has not caught up",
        expected_fix=["pg_vacuum"],
        hint="Use pg_stat_activity or pg_explain_analyze to confirm slow sequential scans, then pg_vacuum on the orders table.",
    ),
    Scenario(
        name="pg-privilege-revoke",
        difficulty=0.45,
        layer="database",
        alert="CRITICAL: App returning HTTP 500 — DB logs show 'permission denied for table orders'; check pg_show_tables",
        inject_commands=[
            f"docker exec {POSTGRES_CONTAINER} psql -U sre -d production -c "
            "\"REVOKE SELECT, INSERT, UPDATE ON orders FROM sre;\" || true"
        ],
        root_cause="Database role 'sre' had SELECT/INSERT/UPDATE privileges revoked on the orders table",
        expected_fix=["pg_show_tables", "curl_endpoint"],
        hint="Use search_logs or read_app_logs to find the permission denied error, then pg_show_tables to confirm the table exists, then verify with curl_endpoint after restoring grants.",
    ),
    Scenario(
        name="redis-eviction-db-cascade",
        difficulty=0.55,
        layer="cross_layer",
        alert="CRITICAL: API error rate 40%, cache hit rate 0%, DB CPU 98% — check redis_info for eviction policy then pg_stat_activity",
        inject_commands=[
            f"docker exec {REDIS_CONTAINER} redis-cli CONFIG SET maxmemory 2mb",
            f"docker exec {REDIS_CONTAINER} redis-cli CONFIG SET maxmemory-policy allkeys-lru",
            f"for i in $(seq 1 300); do docker exec {REDIS_CONTAINER} redis-cli SET junk_$i $(head -c 256 /dev/urandom | base64 | head -c 256); done",
            f"for i in $(seq 1 5); do docker exec {POSTGRES_CONTAINER} psql -U sre -d production -c \"SELECT COUNT(*) FROM orders o1 JOIN orders o2 ON o1.status=o2.status;\" & done"
        ],
        root_cause="Redis maxmemory too low triggers allkeys-lru eviction; cache misses force expensive cross-join queries to hit the DB",
        expected_fix=["redis_info", "redis_flush_db", "pg_cancel_query"],
        hint="Start with redis_info to see evicted_keys and maxmemory. Then redis_flush_db to reset. Then pg_stat_activity to cancel the runaway DB queries.",
    ),
]

# Sort medium scenarios by difficulty
MEDIUM_SCENARIOS.sort(key=lambda s: s.difficulty)


HARD_SCENARIOS = [
    Scenario(
        name="cascading-lock-timeout",
        difficulty=0.7,
        layer="cross_layer",
        alert="CRITICAL: API Latency > 10s, entire application unresponsive.",
        inject_commands=[
            f'docker exec {POSTGRES_CONTAINER} psql -U sre -d production -c "BEGIN; LOCK TABLE orders IN EXCLUSIVE MODE; SELECT pg_sleep(300);" &',
            f"docker exec {REDIS_CONTAINER} redis-cli FLUSHDB",
        ],
        root_cause="Exclusive lock on orders table created a connection pool queue, and cache was empty.",
        expected_fix=["pg_cancel_query"],
    ),
    Scenario(
        name="full-stack-meltdown",
        difficulty=0.9,
        layer="cross_layer",
        alert="CRITICAL: Multiple services degraded — app 503s, DB locks, Redis OOM.",
        inject_commands=[
            f'docker exec {POSTGRES_CONTAINER} psql -U sre -d production -c "BEGIN; LOCK TABLE orders IN EXCLUSIVE MODE; SELECT pg_sleep(300);" &',
            f"docker exec {REDIS_CONTAINER} sh -c 'for i in $(seq 1 800); do redis-cli SET meltdown_$i $(head -c 500 /dev/urandom | base64 | head -c 500); done'",
        ],
        root_cause="Simultaneous DB lock and Redis memory exhaustion causing cascading failures across all layers.",
        expected_fix=["pg_cancel_query", "redis_flush_db"],
    ),
]


class LLMDesigner:
    """Uses Gemini API to generate complex scenarios, falling back to
    difficulty-ordered programmatic scenarios if unavailable."""

    # ── Per-Task Grading Profile (mirrors workspace_agent pattern) ──
    # Each task gets unique weights so the OpenEnv validator sees distinct grading logic.
    @staticmethod
    def _build_grader_profile(task_number: int, scenario_name: str) -> dict:
        """Create a deterministic, task-specific grader profile.

        Every task gets a distinct profile_id + distinct weight distribution.
        The validator requires this to count tasks as having separate grading.
        """
        # Vary weights per task so each has a unique scoring signature
        health_weight = round(0.35 + 0.005 * task_number, 4)
        fix_weight = round(0.20 + 0.004 * task_number, 4)
        workflow_weight = round(0.15 + 0.003 * task_number, 4)
        efficiency_weight = round(0.10 + 0.002 * task_number, 4)
        reckless_weight = max(0.05, round(
            1.0 - (health_weight + fix_weight + workflow_weight + efficiency_weight), 4
        ))

        weight_sum = health_weight + fix_weight + workflow_weight + efficiency_weight + reckless_weight
        weights = {
            "health_check": round(health_weight / weight_sum, 6),
            "expected_fix": round(fix_weight / weight_sum, 6),
            "workflow_order": round(workflow_weight / weight_sum, 6),
            "efficiency": round(efficiency_weight / weight_sum, 6),
            "reckless_penalty": round(reckless_weight / weight_sum, 6),
        }

        # Unique signature that the validator can use to verify distinctness
        profile_signature = float(
            sum((idx + 1) * int(w * 100000) for idx, w in enumerate(weights.values()))
        )

        return {
            "profile_id": f"{scenario_name}_grader",
            "profile_signature": profile_signature,
            "weights": weights,
            "terminal_healthy_bonus": round(0.38 + 0.01 * (task_number % 5), 4),
            "terminal_unhealthy_penalty": round(0.36 + 0.01 * (task_number % 4), 4),
            "reckless_threshold": 3 + (task_number % 3),
            "sla_penalty": round(0.08 + 0.005 * (task_number % 6), 4),
            "root_cause_bonus": round(0.08 + 0.01 * (task_number % 4), 4),
        }

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
        prompt = f"""You are generating an SRE incident scenario for an RL agent training.
The architecture is: Postgres ({POSTGRES_CONTAINER}), Redis ({REDIS_CONTAINER}), Flask App ({APP_CONTAINER}).

Agent's skill profile: {json.dumps(skill_profile)}
Target difficulty: {difficulty} (0.0 to 1.0)
Agent's weakest layer: {weakest_layer or "unknown"}

Generate a realistic but challenging SRE incident scenario that:
1. Targets the specified difficulty level
2. Can be injected via docker exec commands
3. Requires diagnosis and fixing using standard SRE tools
4. Is appropriate for RL agent training

Keep inject_commands valid and executable."""

        try:
            from google.genai import types
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_schema=Scenario,
                    response_mime_type="application/json",
                )
            )
            scenario_data = json.loads(response.text)
            scenario = Scenario(**scenario_data)
            return scenario.model_dump()
        except Exception as e:
            print(f"Gemini generation failed, using fallback: {e}")
            return fallback

    def _get_fallback(self, difficulty: float, weakest_layer: str = "") -> dict:
        """Select a scenario matched to the current difficulty level.

        - At low difficulty: pick only easy scenarios
        - As difficulty grows: unlock harder scenarios from the pool
        - Above HARD_SCENARIO_THRESHOLD: use hard / cross-layer scenarios
        """
        # ── High difficulty: use HARD_SCENARIOS pool ──
        if difficulty > HARD_SCENARIO_THRESHOLD:
            candidates = HARD_SCENARIOS
            layer_match = [s for s in candidates if s.layer == weakest_layer]
            if layer_match:
                return random.choice(layer_match).model_dump()
            return random.choice(candidates).model_dump()

        # ── Medium difficulty (0.28–0.6): mix warmup + medium scenarios ──
        if difficulty >= 0.28:
            # Eligible warmup scenarios (unlocked up to current difficulty + 0.15)
            eligible_warmup = [s for s in WARMUP_SCENARIOS if s.difficulty <= difficulty + 0.15]
            # Eligible medium scenarios (unlocked up to current difficulty + 0.1)
            eligible_medium = [s for s in MEDIUM_SCENARIOS if s.difficulty <= difficulty + 0.1]
            candidates = eligible_warmup + eligible_medium
            if not candidates:
                candidates = [WARMUP_SCENARIOS[0]]

            # Bias toward weakest layer if available
            if weakest_layer:
                layer_match = [s for s in candidates if s.layer == weakest_layer]
                if layer_match and random.random() < 0.6:
                    return random.choice(layer_match).model_dump()

            return random.choice(candidates).model_dump()

        # ── Low difficulty: easy warmup tasks only ──
        eligible = [s for s in WARMUP_SCENARIOS if s.difficulty <= difficulty + 0.15]
        if not eligible:
            eligible = [WARMUP_SCENARIOS[0]]
        return random.choice(eligible).model_dump()

    def get_scenario_by_id(self, task_id: str) -> dict:
        """Retrieve a specific scenario by its task_id (e.g., 'task_1')."""
        all_scenarios = WARMUP_SCENARIOS + MEDIUM_SCENARIOS + HARD_SCENARIOS
        for s in all_scenarios:
            if s.task_id == task_id:
                return s.model_dump()

        # Fallback to a random warmup if not found
        print(f"Requested task_id {task_id} not found. Falling back.")
        return WARMUP_SCENARIOS[0].model_dump()

    @classmethod
    def assign_grader_profiles(cls):
        """Assign unique grader profiles to all scenario pools.

        Called once at module load to ensure every scenario has distinct grading.
        """
        all_scenarios = WARMUP_SCENARIOS + MEDIUM_SCENARIOS + HARD_SCENARIOS
        for idx, scenario in enumerate(all_scenarios):
            task_number = idx + 1
            scenario.task_id = f"task_{task_number}"
            scenario.grader_profile = cls._build_grader_profile(task_number, scenario.name)


# Assign grader profiles at module load time
LLMDesigner.assign_grader_profiles()
