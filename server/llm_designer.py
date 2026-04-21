import json
import random

WARMUP_SCENARIOS = [
    {
        "name": "runaway-query",
        "difficulty": 0.1,
        "layer": "database",
        "alert": "CRITICAL: API p99 latency > 5s, PostgreSQL CPU at 95%",
        "inject_commands": [
            'docker exec pagezero-postgres-1 psql -U sre -d production -c "SELECT pg_sleep(300), * FROM orders o1 CROSS JOIN orders o2 LIMIT 1;" &'
        ],
        "root_cause": "Runaway CROSS JOIN query consuming all CPU",
        "expected_fix": ["pg_cancel_query"]
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
        "expected_fix": ["pg_create_index"]
    },
    {
        "name": "redis-oom",
        "difficulty": 0.2,
        "layer": "cache",
        "alert": "CRITICAL: Redis memory usage > 95%, OOM errors in app logs",
        "inject_commands": [
            "docker exec pagezero-redis-1 redis-cli DEBUG SET-ACTIVE-EXPIRE 0",
            "for i in $(seq 1 1000); do docker exec pagezero-redis-1 redis-cli SET garbage_$i $(head -c 1000 /dev/urandom | base64); done"
        ],
        "root_cause": "Redis memory exhausted by orphaned keys",
        "expected_fix": ["redis_flush_db"]
    },
    {
        "name": "redis-cache-drop",
        "difficulty": 0.3,
        "layer": "cache",
        "alert": "WARNING: Cache miss rate spikes to 100%",
        "inject_commands": [
            "docker exec pagezero-redis-1 redis-cli FLUSHALL"
        ],
        "root_cause": "Cache was accidently dropped, rebuilding data slowly",
        "expected_fix": ["curl_endpoint"]
    },
    {
        "name": "connection-pool-exhausted",
        "difficulty": 0.4,
        "layer": "database",
        "alert": "CRITICAL: 'too many connections' errors, new requests timing out",
        "inject_commands": [
            "for i in $(seq 1 50); do docker exec pagezero-postgres-1 psql -U sre -d production -c 'SELECT pg_sleep(600)' & done"
        ],
        "root_cause": "Leaked connections exhausting PostgreSQL max_connections",
        "expected_fix": ["pg_cancel_query"]
    }
]

class LLMDesigner:
    """Uses an LLM (mocked here, should be connected to OpenAI/Gemini) to generate complex scenarios"""
    
    def __init__(self):
        pass
        
    def design(self, skill_profile: dict, difficulty: float) -> dict:
        # In a real environment, this calls Gemini or Claude.
        # For hackathon robustness (if API breaks), we return a programmatic complex scenario.
        
        # Hard fallback for high-difficulty
        if difficulty > 0.6:
            return {
                "name": "cascading-lock-timeout",
                "difficulty": difficulty,
                "layer": "cross_layer",
                "alert": "CRITICAL: API Latency > 10s, entire application unresponsive.",
                "inject_commands": [
                    # DB Lock
                    'docker exec pagezero-postgres-1 psql -U sre -d production -c "BEGIN; LOCK TABLE orders IN EXCLUSIVE MODE; SELECT pg_sleep(300);" &',
                    # Redis flush (cache cold) to force queries to hit locked DB
                    'docker exec pagezero-redis-1 redis-cli FLUSHDB'
                ],
                "root_cause": "Exclusive lock on orders table created a connection pool queue, and cache was empty.",
                "expected_fix": ["pg_cancel_query"]
            }
            
        # Hard fallback for medium
        return random.choice(WARMUP_SCENARIOS)
