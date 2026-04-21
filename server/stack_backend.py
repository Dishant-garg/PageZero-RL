import subprocess
import json
import time
from typing import Dict, Any

class StackBackend:
    """Executes real commands against Postgres, Redis, and Docker containers."""

    def __init__(self):
        self.incident_start_time = time.time()
        self.revenue_rate_per_minute = 3900.0  # $3,900/min during peak times
        self.sla_threshold_minutes = 5.0

    def reset_incident_timer(self):
        self.incident_start_time = time.time()

    def reset_containers(self):
        """Resets the Docker containers back to a clean state if needed."""
        self._run_cmd("docker compose -f ../docker-compose.yml restart")

    # ═══ PostgreSQL ═══
    def pg_stat_activity(self) -> str:
        return self._run_psql(
            "SELECT pid, state, wait_event_type, extract(epoch from (now() - query_start)) as duration_sec, "
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
        # Simplistic mock since pg_stat_statements requires extension installation
        return self._run_psql(
            "SELECT query, calls, total_exec_time, rows FROM pg_stat_statements ORDER BY total_exec_time DESC LIMIT 5;"
        )

    def pg_cancel_query(self, pid: int) -> str:
        return self._run_psql(f"SELECT pg_cancel_backend({pid});")

    def pg_create_index(self, table: str, column: str) -> str:
        idx_name = f"idx_{table}_{column}"
        return self._run_psql(
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {idx_name} ON {table}({column});"
        )
        
    def pg_vacuum(self, table: str) -> str:
        return self._run_psql(f"VACUUM ANALYZE {table};")
        
    def pg_show_tables(self) -> str:
        return self._run_psql(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='public';"
        )

    # ═══ Redis ═══
    def redis_info(self) -> str:
        return self._run_redis("INFO stats\\r\\nINFO memory")

    def redis_slowlog(self) -> str:
        return self._run_redis("SLOWLOG GET 10")

    def redis_keys(self, pattern: str = "*") -> str:
        return self._run_redis(f"KEYS {pattern}")

    def redis_flush_db(self) -> str:
        return self._run_redis("FLUSHDB")
        
    def redis_get_key(self, key: str) -> str:
        val = self._run_redis(f"GET {key}")
        ttl = self._run_redis(f"TTL {key}")
        return f"Value: {val}\\nTTL: {ttl}"

    # ═══ Docker / Infrastructure ═══
    def docker_ps(self) -> str:
        out = self._run_cmd("docker ps --format '{{.Names}} - {{.Status}}'")
        if out.startswith("ERROR"):
            return "ERROR: Docker daemon is returning an error."
        return out

    def docker_stats(self, container: str) -> str:
        return self._run_cmd(
            f"docker stats {container} --no-stream --format 'CPU: {{.CPUPerc}} MEM: {{.MemUsage}}'"
        )

    def docker_restart(self, container: str) -> str:
        return self._run_cmd(f"docker restart {container}")
        
    def docker_logs(self, container: str, lines: int = 50) -> str:
        return self._run_cmd(f"docker logs --tail={lines} {container}")
        
    def check_disk_usage(self) -> str:
        return self._run_cmd("df -h /")

    # ═══ Application / Alerts ═══
    def curl_endpoint(self, url: str) -> str:
        return self._run_cmd(f"curl -s -m 5 -w '\\nHTTP %{{http_code}} Time: %{{time_total}}s' {url}")

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
            if locks and "0 rows" not in locks and locks.strip() != "":
                # If there's output and it's not empty/0 rows, we have locks
                lines = [l for l in locks.split('\\n') if l.strip() and not l.startswith('(') and not l.startswith('-')]
                if len(lines) > 2: # Has headers and some lock rows
                    return False
            
            return True
        except Exception:
            return False

    # ═══ Internals ═══
    def _run_psql(self, sql: str) -> str:
        # Escape quotes for bash execution
        sql_escaped = sql.replace('"', '\\"')
        result = subprocess.run(
            f'docker exec pagezero-postgres-1 psql -U sre -d production -c "{sql_escaped}"',
            shell=True, capture_output=True, text=True, timeout=15
        )
        if result.returncode != 0:
            return f"ERROR: {result.stderr[:500]}"
        return result.stdout[:2000]

    def _run_redis(self, cmd: str) -> str:
        result = subprocess.run(
            f'docker exec pagezero-redis-1 redis-cli {cmd}',
            shell=True, capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return f"ERROR: {result.stderr[:500]}"
        return result.stdout[:2000]

    def _run_cmd(self, cmd: str) -> str:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=15
        )
        if result.returncode != 0:
            return f"ERROR: {result.stderr[:500]}"
        return result.stdout[:2000]
