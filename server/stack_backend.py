import subprocess
import json
import time
import os
import re
from pathlib import Path
from typing import Dict, Any

from .config import (
    POSTGRES_CONTAINER, REDIS_CONTAINER, APP_CONTAINER,
    REVENUE_RATE_PER_MINUTE, SLA_THRESHOLD_MINUTES,
    APP_HEALTH_URL,
)

from dotenv import load_dotenv
load_dotenv()

# Absolute path to docker-compose.yml for reliable subprocess calls
_REPO_ROOT = Path(__file__).parent.parent
COMPOSE_FILE = str(_REPO_ROOT / "docker-compose.yml")

# Regex to reject anything that isn't a valid SQL identifier
_SAFE_IDENTIFIER = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

# ═══ Remote VM Configuration ═══
# When running on HF Spaces, point to containers on remote VM
# When running locally, these default to localhost (docker containers)
VM_HOST = os.getenv("VM_HOST", "localhost")  # VM IP or hostname
VM_USER = os.getenv("VM_USER", "ubuntu")  # SSH user for VM
VM_SSH_KEY = os.getenv("VM_SSH_KEY", None)  # Path to SSH private key


class StackBackend:
    """Executes real commands against Postgres, Redis, and Docker containers via docker exec."""

    def __init__(self):
        self.incident_start_time = time.time()
        self.revenue_rate_per_minute = REVENUE_RATE_PER_MINUTE
        self.sla_threshold_minutes = SLA_THRESHOLD_MINUTES

    def reset_incident_timer(self):
        self.incident_start_time = time.time()

    def reset_containers(self):
        """Resets the Docker containers back to a clean state."""
        self._run_cmd(f"docker compose -f {COMPOSE_FILE} restart")

    def cleanup_postgres(self):
        """Kill all non-idle client backends and restore permissions."""
        self._run_psql(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
            "WHERE pid <> pg_backend_pid() "
            "AND backend_type = 'client backend' "
            "AND state <> 'idle';"
        )
        # Re-GRANT permissions in case pg-privilege-revoke scenario ran
        self._run_psql(
            "GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO sre;"
        )

    def cleanup_redis(self):
        """Flush Redis, reset memory config, and re-enable active expiry."""
        self._run_redis_cmd("FLUSHDB")
        self._run_redis_cmd("CONFIG", "SET", "maxmemory", "50mb")
        self._run_redis_cmd("CONFIG", "SET", "maxmemory-policy", "noeviction")
        self._run_redis_cmd("DEBUG", "SET-ACTIVE-EXPIRE", "1")

    def revert_schema_drift(self):
        """Revert known schema drifts so the next episode starts clean."""
        # Try to rename email_address back to user_email (idempotent: errors are fine)
        result = self._run_psql(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'orders' AND column_name = 'email_address';"
        )
        if "email_address" in result:
            self._run_psql(
                "ALTER TABLE orders RENAME COLUMN email_address TO user_email;"
            )

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
        clean = query.strip()
        if not clean.upper().startswith("SELECT"):
            return "ERROR: Only SELECT queries allowed for EXPLAIN ANALYZE"
        # Reject multi-statement injection (semicolons, comments)
        if ";" in clean or "--" in clean:
            return "ERROR: Query must be a single SELECT statement (no semicolons or comments)"
        return self._run_psql(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT) {clean}")

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
        # Strict identifier validation
        if not _SAFE_IDENTIFIER.match(table):
            return f"ERROR: Invalid table name '{table}'"
        if not _SAFE_IDENTIFIER.match(column):
            return f"ERROR: Invalid column name '{column}'"
        idx_name = f"idx_{table}_{column}"
        return self._run_psql(
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {idx_name} ON {table}({column});"
        )

    def pg_vacuum(self, table: str) -> str:
        if not _SAFE_IDENTIFIER.match(table):
            return f"ERROR: Invalid table name '{table}'"
        return self._run_psql(f"VACUUM ANALYZE {table};")

    def pg_show_tables(self) -> str:
        return self._run_psql(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='public';"
        )

    # ═══ Redis ═══
    def redis_info(self) -> str:
        """Get Redis server stats and memory info (two separate redis-cli calls)."""
        stats = self._run_redis_cmd("INFO", "stats")
        memory = self._run_redis_cmd("INFO", "memory")
        return f"=== STATS ===\n{stats}\n\n=== MEMORY ===\n{memory}"[:2000]

    def redis_slowlog(self) -> str:
        return self._run_redis_cmd("SLOWLOG", "GET", "10")

    def redis_keys(self, pattern: str = "*") -> str:
        pattern = pattern.strip() or "*"
        return self._run_redis_cmd("KEYS", pattern)

    def redis_flush_db(self) -> str:
        return self._run_redis_cmd("FLUSHDB")

    def redis_get_key(self, key: str) -> str:
        val = self._run_redis_cmd("GET", key)
        ttl = self._run_redis_cmd("TTL", key)
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
        """Verifies if the actual stack is healthy — checks App, DB, and Redis."""
        try:
            # 1. Check App
            app_health = self.curl_endpoint(APP_HEALTH_URL)
            if "HTTP 200" not in app_health:
                return False

            # 2. Check DB locks
            locks = self.pg_locks()
            if locks and "ERROR" not in locks:
                data_lines = [
                    l for l in locks.split("\n")
                    if l.strip()
                    and not l.startswith("-")
                    and not l.startswith("(")
                    and "blocked_pid" not in l.lower()
                ]
                if len(data_lines) > 0:
                    return False

            # 3. Check Redis is reachable and not OOM
            redis_mem = self._run_redis_cmd("INFO", "memory")
            if "ERROR" in redis_mem:
                return False
            # Check if used_memory is near maxmemory (>90%)
            used = 0
            maxm = 0
            for line in redis_mem.split("\n"):
                if line.startswith("used_memory:"):
                    used = int(line.split(":")[1].strip())
                elif line.startswith("maxmemory:"):
                    maxm = int(line.split(":")[1].strip())
            if maxm > 0 and used > maxm * 0.9:
                return False

            return True
        except Exception:
            return False

    # ═══ Internals ═══
    def _run_psql(self, sql: str) -> str:
        """Run a SQL command inside postgres container (local or remote via SSH)."""
        docker_cmd = (
            f"docker exec {POSTGRES_CONTAINER} "
            f"psql -U sre -d production -c \"{sql}\""
        )
        
        if VM_HOST == "localhost":
            result = subprocess.run(
                docker_cmd, shell=True, capture_output=True, text=True, timeout=15
            )
        else:
            ssh_args = ["ssh", "-q", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null"]
            if VM_SSH_KEY:
                ssh_args.extend(["-i", VM_SSH_KEY])
            ssh_args.extend([f"{VM_USER}@{VM_HOST}", docker_cmd])
            result = subprocess.run(
                ssh_args, shell=False, capture_output=True, text=True, timeout=15
            )
        if result.returncode != 0:
            return f"ERROR: {result.stderr[:500]}"
        return result.stdout[:2000]

    def _run_redis_cmd(self, *args: str) -> str:
        """Run a Redis command inside redis container (local or remote via SSH)."""
        args_str = " ".join(args)
        docker_cmd = f"docker exec {REDIS_CONTAINER} redis-cli {args_str}"
        
        if VM_HOST == "localhost":
            result = subprocess.run(
                docker_cmd, shell=True, capture_output=True, text=True, timeout=10
            )
        else:
            ssh_args = ["ssh", "-q", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null"]
            if VM_SSH_KEY:
                ssh_args.extend(["-i", VM_SSH_KEY])
            ssh_args.extend([f"{VM_USER}@{VM_HOST}", docker_cmd])
            result = subprocess.run(
                ssh_args, shell=False, capture_output=True, text=True, timeout=10
            )
        if result.returncode != 0:
            return f"ERROR: {result.stderr[:500]}"
        return result.stdout[:2000]

    # Keep backward-compatible alias
    def _run_redis(self, cmd: str) -> str:
        """Legacy alias — splits cmd string into args for _run_redis_cmd."""
        return self._run_redis_cmd(*cmd.split())

    def _run_cmd(self, cmd: str) -> str:
        """Run an arbitrary shell command (local or remote via SSH to VM)."""
        if VM_HOST == "localhost":
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=15
            )
        else:
            ssh_args = ["ssh", "-q", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null"]
            if VM_SSH_KEY:
                ssh_args.extend(["-i", VM_SSH_KEY])
            ssh_args.extend([f"{VM_USER}@{VM_HOST}", cmd])
            result = subprocess.run(
                ssh_args, shell=False, capture_output=True, text=True, timeout=15
            )
        if result.returncode != 0:
            return f"ERROR: {result.stderr[:500]}"
        return result.stdout[:2000]
