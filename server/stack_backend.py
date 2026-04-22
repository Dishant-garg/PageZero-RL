import subprocess
import json
import time
import os
from pathlib import Path
from typing import Dict, Any

# Absolute path to docker-compose.yml for reliable subprocess calls
_REPO_ROOT = Path(__file__).parent.parent
COMPOSE_FILE = str(_REPO_ROOT / "docker-compose.yml")

class StackBackend:
    """Executes real commands against Postgres, Redis, and Docker containers via docker exec."""

    def __init__(self):
        self.incident_start_time = time.time()
        self.revenue_rate_per_minute = 3900.0  # $3,900/min during peak times
        self.sla_threshold_minutes = 5.0

    def reset_incident_timer(self):
        self.incident_start_time = time.time()

    def reset_containers(self):
        """Resets the Docker containers back to a clean state."""
        self._run_cmd(f"docker compose -f {COMPOSE_FILE} restart")

    # ═══ PostgreSQL ═══
    def pg_stat_activity(self) -> str:
        return self._run_psql(
            "SELECT pid, state, wait_event_type, "
            "extract(epoch from (now() - query_start)) as duration_sec, "
            "LEFT(query, 120) as query FROM pg_stat_activity "
            "WHERE state != 'idle' AND backend_type = 'client backend' "
            "ORDER BY query_start LIMIT 20;"
        )

    def pg_locks(self) -> str:
        return self._run_psql(
            "SELECT blocked.pid AS blocked_pid, "
            "blocking.pid AS blocking_pid, "
            "LEFT(blocked.query, 60) AS blocked_query "
            "FROM pg_locks blocked_locks "
            "JOIN pg_stat_activity blocked ON blocked.pid = blocked_locks.pid "
            "JOIN pg_locks blocking_locks ON blocking_locks.locktype = blocked_locks.locktype "
            "JOIN pg_stat_activity blocking ON blocking.pid = blocking_locks.pid "
            "WHERE NOT blocked_locks.granted LIMIT 10;"
        )

    def pg_explain_analyze(self, query: str) -> str:
        if not query.strip().upper().startswith("SELECT"):
            return "ERROR: Only SELECT queries allowed for EXPLAIN ANALYZE"
        return self._run_psql(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT) {query}")

    def pg_stat_statements(self) -> str:
        # Check if pg_stat_statements extension is available
        check = self._run_psql(
            "SELECT query, calls, total_exec_time, rows "
            "FROM pg_stat_statements ORDER BY total_exec_time DESC LIMIT 5;"
        )
        if "ERROR" in check:
            # Fallback: show slow queries from pg_stat_activity
            return self._run_psql(
                "SELECT pid, state, extract(epoch from (now() - query_start)) as sec, "
                "LEFT(query, 100) as query FROM pg_stat_activity "
                "WHERE state != 'idle' ORDER BY query_start LIMIT 10;"
            )
        return check

    def pg_cancel_query(self, pid: int) -> str:
        return self._run_psql(f"SELECT pg_cancel_backend({int(pid)});")

    def pg_create_index(self, table: str, column: str) -> str:
        # Sanitize names to prevent SQLi
        table = table.replace('"', "").replace(";", "")
        column = column.replace('"', "").replace(";", "")
        idx_name = f"idx_{table}_{column}"
        return self._run_psql(
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {idx_name} ON {table}({column});"
        )

    def pg_vacuum(self, table: str) -> str:
        table = table.replace('"', "").replace(";", "")
        return self._run_psql(f"VACUUM ANALYZE {table};")

    def pg_show_tables(self) -> str:
        return self._run_psql(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='public';"
        )

    # ═══ Redis ═══
    def redis_info(self) -> str:
        """Get Redis server stats and memory info (two separate redis-cli calls)."""
        stats = self._run_redis_cmd("INFO stats")
        memory = self._run_redis_cmd("INFO memory")
        return f"=== STATS ===\n{stats}\n\n=== MEMORY ===\n{memory}"[:2000]

    def redis_slowlog(self) -> str:
        return self._run_redis_cmd("SLOWLOG GET 10")

    def redis_keys(self, pattern: str = "*") -> str:
        # Safety: don't allow dangerous patterns
        pattern = pattern.strip() or "*"
        return self._run_redis_cmd(f"KEYS {pattern}")

    def redis_flush_db(self) -> str:
        return self._run_redis_cmd("FLUSHDB")

    def redis_get_key(self, key: str) -> str:
        val = self._run_redis_cmd(f"GET {key}")
        ttl = self._run_redis_cmd(f"TTL {key}")
        return f"Value: {val}\nTTL: {ttl}"

    # ═══ Docker / Infrastructure ═══
    def docker_ps(self) -> str:
        out = self._run_cmd("docker ps --format '{{.Names}} - {{.Status}}'")
        if out.startswith("ERROR"):
            return "ERROR: Docker daemon is returning an error."
        return out

    def docker_stats(self, container: str) -> str:
        return self._run_cmd(
            f"docker stats {container} --no-stream --format 'CPU: {{{{.CPUPerc}}}} MEM: {{{{.MemUsage}}}}'"
        )

    def docker_restart(self, container: str) -> str:
        return self._run_cmd(f"docker restart {container}")

    def docker_logs(self, container: str, lines: int = 50) -> str:
        return self._run_cmd(f"docker logs --tail={int(lines)} {container} 2>&1")

    def check_disk_usage(self) -> str:
        return self._run_cmd("df -h /")

    # ═══ Application / Alerts ═══
    def curl_endpoint(self, url: str) -> str:
        return self._run_cmd(
            f"curl -s -m 5 -w '\\nHTTP %{{http_code}} Time: %{{time_total}}s' '{url}'"
        )

    # ═══ SLA Tracking ═══
    def get_sla_status(self) -> Dict[str, Any]:
        elapsed_sec = time.time() - self.incident_start_time
        elapsed_min = elapsed_sec / 60.0
        return {
            "downtime_minutes": round(elapsed_min, 1),
            "revenue_loss_usd": round(elapsed_min * self.revenue_rate_per_minute, 2),
            "sla_status": "OK" if elapsed_min < self.sla_threshold_minutes else "VIOLATED",
        }

    # ═══ Verifier ═══
    def verify_resolution(self) -> bool:
        """Verifies if the actual stack is healthy."""
        try:
            # Check App
            app_health = self.curl_endpoint("http://localhost:5000/health")
            if "HTTP 200" not in app_health:
                return False

            # Check DB locks
            locks = self.pg_locks()
            if locks and "ERROR" not in locks:
                # Count lock rows (skip header lines)
                data_lines = [
                    l for l in locks.split("\n")
                    if l.strip()
                    and not l.startswith("-")
                    and not l.startswith("(")
                    and "blocked_pid" not in l.lower()
                ]
                if len(data_lines) > 0:
                    return False

            return True
        except Exception:
            return False

    # ═══ Internals ═══
    def _run_psql(self, sql: str) -> str:
        """Run a SQL command inside the postgres container."""
        result = subprocess.run(
            ["docker", "exec", "pagezero-postgres-1",
             "psql", "-U", "sre", "-d", "production", "-c", sql],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode != 0:
            return f"ERROR: {result.stderr[:500]}"
        return result.stdout[:2000]

    def _run_redis_cmd(self, cmd: str) -> str:
        """Run a single Redis command inside the redis container."""
        result = subprocess.run(
            f"docker exec pagezero-redis-1 redis-cli {cmd}",
            shell=True, capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return f"ERROR: {result.stderr[:500]}"
        return result.stdout[:2000]

    # Keep backward-compatible alias
    def _run_redis(self, cmd: str) -> str:
        return self._run_redis_cmd(cmd)

    def _run_cmd(self, cmd: str) -> str:
        """Run an arbitrary shell command on the host."""
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=15
        )
        if result.returncode != 0:
            return f"ERROR: {result.stderr[:500]}"
        return result.stdout[:2000]
