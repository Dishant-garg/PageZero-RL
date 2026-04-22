import traceback
from typing import Dict, Any
from .stack_backend import StackBackend
from .config import APP_CONTAINER, POSTGRES_CONTAINER, REDIS_CONTAINER


class Executor:
    def __init__(self, backend: StackBackend):
        self.backend = backend

    def execute(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Routes generic tool calls to specific backend methods safely."""
        try:
            # Monitoring & Triage
            if tool_name == "check_alerts":
                ps = self.backend.docker_ps()
                app_health = self.backend.curl_endpoint("http://localhost:5001/health")
                return f"Container Status:\n{ps}\n\nApp Health:\n{app_health}"
            elif tool_name == "get_service_metrics":
                service = args.get("service")
                if not service:
                    return "ERROR: missing 'service' argument"
                return self.backend.docker_stats(f"pagezero-{service}-1")
            elif tool_name == "get_error_rate":
                logs = self.backend.docker_logs(APP_CONTAINER, lines=30)
                error_lines = [
                    l for l in logs.split("\n")
                    if "ERROR" in l or "500" in l or "exception" in l.lower()
                ]
                if error_lines:
                    return (
                        f"Recent errors ({len(error_lines)} lines):\n"
                        + "\n".join(error_lines[-10:])
                    )
                return "No recent errors detected in app logs."

            # Application
            elif tool_name == "read_app_logs":
                service = args.get("service", "app")
                lines = args.get("lines", 50)
                return self.backend.docker_logs(f"pagezero-{service}-1", lines)
            elif tool_name == "search_logs":
                pattern = args.get("pattern", "")
                if not pattern:
                    return "ERROR: missing 'pattern'"
                app_logs = self.backend.docker_logs(APP_CONTAINER, lines=200)
                pg_logs = self.backend.docker_logs(POSTGRES_CONTAINER, lines=100)
                combined = app_logs + "\n" + pg_logs
                hits = [l for l in combined.split("\n") if pattern.lower() in l.lower()]
                if not hits:
                    return f"Pattern '{pattern}' not found in recent logs."
                return (
                    f"Found {len(hits)} matches for '{pattern}':\n"
                    + "\n".join(hits[:30])
                )
            elif tool_name == "get_recent_deploys":
                out = self.backend._run_cmd(
                    f"docker inspect {APP_CONTAINER} --format '{{{{.Created}}}} image={{{{.Config.Image}}}}'"
                )
                return f"Last deploy info:\n{out}"
            elif tool_name == "rollback_deploy":
                out = self.backend.docker_restart(APP_CONTAINER)
                return f"Rollback executed (container restarted): {out}"
            elif tool_name == "curl_endpoint":
                url = args.get("url")
                if not url:
                    return "ERROR: missing 'url'"
                return self.backend.curl_endpoint(url)

            # Database
            elif tool_name == "pg_stat_activity":
                return self.backend.pg_stat_activity()
            elif tool_name == "pg_locks":
                return self.backend.pg_locks()
            elif tool_name == "pg_explain_analyze":
                query = args.get("query")
                if not query:
                    return "ERROR: missing 'query'"
                return self.backend.pg_explain_analyze(query)
            elif tool_name == "pg_stat_statements":
                return self.backend.pg_stat_statements()
            elif tool_name == "pg_cancel_query":
                pid = args.get("pid")
                if not pid:
                    return "ERROR: missing 'pid'"
                return self.backend.pg_cancel_query(pid)
            elif tool_name == "pg_create_index":
                table = args.get("table")
                column = args.get("column")
                if not table or not column:
                    return "ERROR: missing 'table' or 'column'"
                return self.backend.pg_create_index(table, column)
            elif tool_name == "pg_vacuum":
                table = args.get("table", "")
                return self.backend.pg_vacuum(table)
            elif tool_name == "pg_show_tables":
                return self.backend.pg_show_tables()

            # Redis
            elif tool_name == "redis_info":
                return self.backend.redis_info()
            elif tool_name == "redis_slowlog":
                return self.backend.redis_slowlog()
            elif tool_name == "redis_keys":
                pattern = args.get("pattern", "*")
                return self.backend.redis_keys(pattern)
            elif tool_name == "redis_get_key":
                key = args.get("key")
                if not key:
                    return "ERROR: missing 'key'"
                return self.backend.redis_get_key(key)
            elif tool_name == "redis_flush_db":
                return self.backend.redis_flush_db()

            # Docker / Infra
            elif tool_name == "docker_ps":
                return self.backend.docker_ps()
            elif tool_name == "docker_stats":
                container = args.get("container")
                if not container:
                    return "ERROR: missing 'container'"
                return self.backend.docker_stats(container)
            elif tool_name == "docker_restart":
                container = args.get("container")
                if not container:
                    return "ERROR: missing 'container'"
                return self.backend.docker_restart(container)
            elif tool_name == "docker_logs":
                container = args.get("container")
                if not container:
                    return "ERROR: missing 'container'"
                lines = args.get("lines", 50)
                return self.backend.docker_logs(container, lines)
            elif tool_name == "check_disk_usage":
                return self.backend.check_disk_usage()

            # Meta / Resolution
            elif tool_name == "diagnose_root_cause":
                cause = args.get("root_cause")
                if not cause:
                    return "ERROR: Please provide a description."
                return f"Root cause logged: {cause}"
            elif tool_name == "done":
                return "Investigation concluded."

            else:
                return f"ERROR: Tool '{tool_name}' not implemented or not available."

        except Exception as e:
            return f"ERROR executing {tool_name}: {str(e)}\n{traceback.format_exc()[:500]}"
