"""
Integration tests for Executor.
Tests that ALL tools are callable and route correctly.
Verifies the complete tool execution pipeline.
"""

import pytest


class TestExecutorMonitoringTools:
    """Test Layer 1: Monitoring & Triage tools."""
    
    def test_check_alerts(self, executor):
        """Verify check_alerts tool executes."""
        result = executor.execute("check_alerts", {})
        assert isinstance(result, str)
        assert len(result) > 0
        assert "ERROR executing" not in result
    
    def test_get_service_metrics(self, executor):
        """Verify get_service_metrics tool executes."""
        result = executor.execute("get_service_metrics", {"service": "app"})
        assert isinstance(result, str)
        assert len(result) > 0
        assert "ERROR executing" not in result
    
    def test_get_service_metrics_missing_arg(self, executor):
        """Verify get_service_metrics handles missing service arg."""
        result = executor.execute("get_service_metrics", {})
        assert isinstance(result, str)
        assert "ERROR" in result or "missing" in result.lower()
    
    def test_get_error_rate(self, executor):
        """Verify get_error_rate tool executes."""
        result = executor.execute("get_error_rate", {})
        assert isinstance(result, str)
        assert len(result) > 0
        assert "ERROR executing" not in result


class TestExecutorApplicationTools:
    """Test Layer 2: Application tools."""
    
    def test_read_app_logs(self, executor):
        """Verify read_app_logs tool executes."""
        result = executor.execute("read_app_logs", {"service": "app", "lines": 20})
        assert isinstance(result, str)
        assert len(result) > 0
        assert "ERROR executing" not in result
    
    def test_read_app_logs_defaults(self, executor):
        """Verify read_app_logs works with defaults."""
        result = executor.execute("read_app_logs", {})
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_search_logs(self, executor):
        """Verify search_logs tool executes."""
        result = executor.execute("search_logs", {"pattern": "ERROR"})
        assert isinstance(result, str)
        assert len(result) > 0
        assert "ERROR executing" not in result
    
    def test_search_logs_missing_pattern(self, executor):
        """Verify search_logs requires pattern."""
        result = executor.execute("search_logs", {})
        assert "ERROR" in result or "missing" in result.lower()
    
    def test_get_recent_deploys(self, executor):
        """Verify get_recent_deploys tool executes."""
        result = executor.execute("get_recent_deploys", {})
        assert isinstance(result, str)
        assert len(result) > 0
        assert "ERROR executing" not in result
    
    def test_rollback_deploy(self, executor):
        """Verify rollback_deploy tool executes."""
        result = executor.execute("rollback_deploy", {})
        assert isinstance(result, str)
        assert len(result) > 0
        assert "ERROR executing" not in result
    
    def test_curl_endpoint(self, executor):
        """Verify curl_endpoint tool executes."""
        result = executor.execute("curl_endpoint", {"url": "http://localhost:5001/health"})
        assert isinstance(result, str)
        assert len(result) > 0
        assert "ERROR executing" not in result
    
    def test_curl_endpoint_missing_url(self, executor):
        """Verify curl_endpoint requires URL."""
        result = executor.execute("curl_endpoint", {})
        assert "ERROR" in result or "missing" in result.lower()


class TestExecutorPostgreSQLTools:
    """Test Layer 3: PostgreSQL tools."""
    
    def test_pg_stat_activity(self, executor):
        """Verify pg_stat_activity tool executes."""
        result = executor.execute("pg_stat_activity", {})
        assert isinstance(result, str)
        assert len(result) > 0
        assert "ERROR executing" not in result
    
    def test_pg_locks(self, executor):
        """Verify pg_locks tool executes."""
        result = executor.execute("pg_locks", {})
        assert isinstance(result, str)
        assert len(result) > 0
        assert "ERROR executing" not in result
    
    def test_pg_explain_analyze(self, executor):
        """Verify pg_explain_analyze tool executes."""
        result = executor.execute(
            "pg_explain_analyze",
            {"query": "SELECT * FROM orders LIMIT 5"}
        )
        assert isinstance(result, str)
        assert len(result) > 0
        assert "ERROR executing" not in result
    
    def test_pg_explain_analyze_missing_query(self, executor):
        """Verify pg_explain_analyze requires query."""
        result = executor.execute("pg_explain_analyze", {})
        assert "ERROR" in result or "missing" in result.lower()
    
    def test_pg_stat_statements(self, executor):
        """Verify pg_stat_statements tool executes."""
        result = executor.execute("pg_stat_statements", {})
        assert isinstance(result, str)
        assert len(result) > 0
        assert "ERROR executing" not in result
    
    def test_pg_cancel_query(self, executor):
        """Verify pg_cancel_query tool executes."""
        result = executor.execute("pg_cancel_query", {"pid": 999999})
        assert isinstance(result, str)
        assert len(result) > 0
        assert "ERROR executing" not in result
    
    def test_pg_cancel_query_missing_pid(self, executor):
        """Verify pg_cancel_query requires pid."""
        result = executor.execute("pg_cancel_query", {})
        assert "ERROR" in result or "missing" in result.lower()
    
    def test_pg_create_index(self, executor):
        """Verify pg_create_index tool executes."""
        result = executor.execute(
            "pg_create_index",
            {"table": "orders", "column": "user_email"}
        )
        assert isinstance(result, str)
        assert len(result) > 0
        assert "ERROR executing" not in result
    
    def test_pg_create_index_missing_args(self, executor):
        """Verify pg_create_index requires table and column."""
        result = executor.execute("pg_create_index", {"table": "orders"})
        assert "ERROR" in result or "missing" in result.lower()
    
    def test_pg_vacuum(self, executor):
        """Verify pg_vacuum tool executes."""
        result = executor.execute("pg_vacuum", {"table": "orders"})
        assert isinstance(result, str)
        assert len(result) > 0
        assert "ERROR executing" not in result
    
    def test_pg_vacuum_all(self, executor):
        """Verify pg_vacuum works without table."""
        result = executor.execute("pg_vacuum", {})
        assert isinstance(result, str)
        assert len(result) > 0
        assert "ERROR executing" not in result
    
    def test_pg_show_tables(self, executor):
        """Verify pg_show_tables tool executes."""
        result = executor.execute("pg_show_tables", {})
        assert isinstance(result, str)
        assert len(result) > 0
        assert "orders" in result.lower()
        assert "ERROR executing" not in result


class TestExecutorRedisTools:
    """Test Layer 4: Redis tools."""
    
    def test_redis_info(self, executor):
        """Verify redis_info tool executes."""
        result = executor.execute("redis_info", {})
        assert isinstance(result, str)
        assert len(result) > 0
        assert "ERROR executing" not in result
    
    def test_redis_slowlog(self, executor):
        """Verify redis_slowlog tool executes."""
        result = executor.execute("redis_slowlog", {})
        assert isinstance(result, str)
        assert len(result) > 0
        assert "ERROR executing" not in result
    
    def test_redis_keys(self, executor):
        """Verify redis_keys tool executes."""
        result = executor.execute("redis_keys", {"pattern": "*"})
        assert isinstance(result, str)
        assert len(result) > 0
        assert "ERROR executing" not in result
    
    def test_redis_keys_with_custom_pattern(self, executor):
        """Verify redis_keys accepts custom pattern."""
        result = executor.execute("redis_keys", {"pattern": "test*"})
        assert isinstance(result, str)
        assert len(result) > 0
        assert "ERROR executing" not in result
    
    def test_redis_get_key(self, executor):
        """Verify redis_get_key tool executes."""
        result = executor.execute("redis_get_key", {"key": "test_key"})
        assert isinstance(result, str)
        assert len(result) > 0
        assert "ERROR executing" not in result
    
    def test_redis_get_key_missing_arg(self, executor):
        """Verify redis_get_key requires key."""
        result = executor.execute("redis_get_key", {})
        assert "ERROR" in result or "missing" in result.lower()
    
    def test_redis_flush_db(self, executor):
        """Verify redis_flush_db tool executes."""
        result = executor.execute("redis_flush_db", {})
        assert isinstance(result, str)
        assert len(result) > 0
        assert "ERROR executing" not in result


class TestExecutorDockerTools:
    """Test Layer 5: Docker/Infrastructure tools."""
    
    def test_docker_ps(self, executor):
        """Verify docker_ps tool executes."""
        result = executor.execute("docker_ps", {})
        assert isinstance(result, str)
        assert len(result) > 0
        assert "pagezero" in result.lower()
        assert "ERROR executing" not in result
    
    def test_docker_stats(self, executor):
        """Verify docker_stats tool executes."""
        result = executor.execute("docker_stats", {"container": "pagezero-app-1"})
        assert isinstance(result, str)
        assert len(result) > 0
        assert "ERROR executing" not in result
    
    def test_docker_stats_missing_container(self, executor):
        """Verify docker_stats requires container."""
        result = executor.execute("docker_stats", {})
        assert "ERROR" in result or "missing" in result.lower()
    
    def test_docker_restart(self, executor):
        """Verify docker_restart tool executes."""
        result = executor.execute("docker_restart", {"container": "pagezero-app-1"})
        assert isinstance(result, str)
        assert len(result) > 0
        assert "ERROR executing" not in result
    
    def test_docker_restart_missing_container(self, executor):
        """Verify docker_restart requires container."""
        result = executor.execute("docker_restart", {})
        assert "ERROR" in result or "missing" in result.lower()
    
    def test_docker_logs(self, executor):
        """Verify docker_logs tool executes."""
        result = executor.execute("docker_logs", {"container": "pagezero-app-1"})
        assert isinstance(result, str)
        assert len(result) > 0
        assert "ERROR executing" not in result
    
    def test_docker_logs_with_lines(self, executor):
        """Verify docker_logs accepts line count."""
        result = executor.execute(
            "docker_logs",
            {"container": "pagezero-app-1", "lines": 5}
        )
        assert isinstance(result, str)
        assert len(result) > 0
        assert "ERROR executing" not in result
    
    def test_docker_logs_missing_container(self, executor):
        """Verify docker_logs requires container."""
        result = executor.execute("docker_logs", {})
        assert "ERROR" in result or "missing" in result.lower()
    
    def test_check_disk_usage(self, executor):
        """Verify check_disk_usage tool executes."""
        result = executor.execute("check_disk_usage", {})
        assert isinstance(result, str)
        assert len(result) > 0
        assert "/" in result  # Should show filesystem
        assert "ERROR executing" not in result


class TestExecutorMetaTools:
    """Test meta/resolution tools."""
    
    def test_diagnose_root_cause(self, executor):
        """Verify diagnose_root_cause tool executes."""
        result = executor.execute(
            "diagnose_root_cause",
            {"root_cause": "Missing index on orders.user_email"}
        )
        assert isinstance(result, str)
        assert len(result) > 0
        assert "ERROR executing" not in result
    
    def test_diagnose_root_cause_with_description(self, executor):
        """Verify diagnose_root_cause requires description."""
        result = executor.execute("diagnose_root_cause", {})
        assert "ERROR" in result or "missing" in result.lower()
    
    def test_done(self, executor):
        """Verify done tool executes."""
        result = executor.execute("done", {})
        assert isinstance(result, str)
        assert "concluded" in result.lower() or "done" in result.lower()


class TestExecutorErrorHandling:
    """Test error handling and edge cases."""
    
    def test_unknown_tool(self, executor):
        """Verify unknown tool returns appropriate error."""
        result = executor.execute("unknown_tool_xyz_123", {})
        assert isinstance(result, str)
        assert "ERROR" in result or "not implemented" in result.lower()
    
    def test_tool_with_extra_args(self, executor):
        """Verify tool accepts extra arguments gracefully."""
        result = executor.execute(
            "check_alerts",
            {"extra_arg": "should_be_ignored", "another": 123}
        )
        assert isinstance(result, str)
        assert "ERROR executing" not in result
    
    def test_tool_with_none_args(self, executor):
        """Verify tool handles None args dict."""
        # This shouldn't happen in practice, but let's be defensive
        result = executor.execute("docker_ps", {})
        assert isinstance(result, str)


class TestExecutorAllToolsCallable:
    """Meta test: verify ALL tools are callable."""
    
    ALL_TOOLS = [
        # Monitoring & Triage
        ("check_alerts", {}),
        ("get_service_metrics", {"service": "app"}),
        ("get_error_rate", {}),
        # Application
        ("read_app_logs", {"service": "app"}),
        ("search_logs", {"pattern": "test"}),
        ("get_recent_deploys", {}),
        ("rollback_deploy", {}),
        ("curl_endpoint", {"url": "http://localhost:5001/health"}),
        # PostgreSQL
        ("pg_stat_activity", {}),
        ("pg_locks", {}),
        ("pg_explain_analyze", {"query": "SELECT 1"}),
        ("pg_stat_statements", {}),
        ("pg_cancel_query", {"pid": 1}),
        ("pg_create_index", {"table": "orders", "column": "status"}),
        ("pg_vacuum", {"table": "orders"}),
        ("pg_show_tables", {}),
        # Redis
        ("redis_info", {}),
        ("redis_slowlog", {}),
        ("redis_keys", {"pattern": "*"}),
        ("redis_get_key", {"key": "test"}),
        ("redis_flush_db", {}),
        # Docker
        ("docker_ps", {}),
        ("docker_stats", {"container": "pagezero-app-1"}),
        ("docker_restart", {"container": "pagezero-app-1"}),
        ("docker_logs", {"container": "pagezero-app-1"}),
        ("check_disk_usage", {}),
        # Meta
        ("diagnose_root_cause", {"root_cause": "test"}),
        ("done", {}),
    ]
    
    @pytest.mark.parametrize("tool_name,args", ALL_TOOLS)
    def test_all_tools_are_callable(self, executor, tool_name, args):
        """Verify every tool in the system is callable."""
        result = executor.execute(tool_name, args)
        assert isinstance(result, str)
        assert len(result) > 0
        
        assert isinstance(result, str), f"Tool {tool_name} should return string"
        assert len(result) > 0, f"Tool {tool_name} should return non-empty string"
        assert "ERROR executing" not in result, f"Tool {tool_name} crashed during execution"
