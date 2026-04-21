import traceback
from typing import Dict, Any
from .stack_backend import StackBackend

class Executor:
    def __init__(self, backend: StackBackend):
        self.backend = backend

    def execute(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Routes generic tool calls to specific backend methods safely."""
        try:
            # Monitoring & Triage
            if tool_name == "check_alerts":
                # Simulated for now, relies on environment telling us active alerts
                return "Active Alerts: Check environment state"
            elif tool_name == "get_service_metrics":
                service = args.get("service")
                if not service: return "ERROR: missing 'service' argument"
                return self.backend.docker_stats(f"pagezero-{service}-1")
            elif tool_name == "get_error_rate":
                return "Error rate data... (simulated)"

            # Application
            elif tool_name == "read_app_logs":
                service = args.get("service", "app")
                lines = args.get("lines", 50)
                return self.backend.docker_logs(f"pagezero-{service}-1", lines)
            elif tool_name == "curl_endpoint":
                url = args.get("url")
                if not url: return "ERROR: missing 'url'"
                return self.backend.curl_endpoint(url)
            
            # Database
            elif tool_name == "pg_stat_activity":
                return self.backend.pg_stat_activity()
            elif tool_name == "pg_locks":
                return self.backend.pg_locks()
            elif tool_name == "pg_explain_analyze":
                query = args.get("query")
                if not query: return "ERROR: missing 'query'"
                return self.backend.pg_explain_analyze(query)
            elif tool_name == "pg_stat_statements":
                return self.backend.pg_stat_statements()
            elif tool_name == "pg_cancel_query":
                pid = args.get("pid")
                if not pid: return "ERROR: missing 'pid'"
                return self.backend.pg_cancel_query(pid)
            elif tool_name == "pg_create_index":
                table = args.get("table")
                column = args.get("column")
                if not table or not column: return "ERROR: missing 'table' or 'column'"
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
                if not key: return "ERROR: missing 'key'"
                return self.backend.redis_get_key(key)
            elif tool_name == "redis_flush_db":
                return self.backend.redis_flush_db()
            
            # Docker / Infra
            elif tool_name == "docker_ps":
                return self.backend.docker_ps()
            elif tool_name == "docker_stats":
                container = args.get("container")
                if not container: return "ERROR: missing 'container'"
                return self.backend.docker_stats(container)
            elif tool_name == "docker_restart":
                container = args.get("container")
                if not container: return "ERROR: missing 'container'"
                return self.backend.docker_restart(container)
            elif tool_name == "docker_logs":
                container = args.get("container")
                if not container: return "ERROR: missing 'container'"
                lines = args.get("lines", 50)
                return self.backend.docker_logs(container, lines)
            elif tool_name == "check_disk_usage":
                return self.backend.check_disk_usage()
            
            # Meta
            elif tool_name == "diagnose_root_cause":
                cause = args.get("root_cause")
                if not cause: return "ERROR: Please provide a description."
                return f"Root cause logged: {cause}"
            elif tool_name == "done":
                return "Investigation concluded."
            
            else:
                return f"ERROR: Tool '{tool_name}' not implemented or not available."
                
        except Exception as e:
            return f"ERROR executing {tool_name}: {str(e)}\\n{traceback.format_exc()}"
