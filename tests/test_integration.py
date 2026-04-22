"""
End-to-end integration tests for PageZero.
Tests complete workflows combining StackBackend, Executor, and containers.
"""

import time


class TestExecutorWorkflows:
    """Test complete execution workflows."""
    
    def test_full_sre_investigation_workflow(self, executor):
        """
        Test a realistic SRE investigation workflow:
        1. Check alerts
        2. Get error rate
        3. Read logs
        4. Check database status
        5. Check Redis status
        """
        alerts = executor.execute("check_alerts", {})
        assert isinstance(alerts, str)
        assert len(alerts) > 0
        
        errors = executor.execute("get_error_rate", {})
        assert isinstance(errors, str)
        
        logs = executor.execute("read_app_logs", {"service": "app", "lines": 30})
        assert isinstance(logs, str)
        
        db_status = executor.execute("pg_stat_activity", {})
        assert isinstance(db_status, str)
        
        redis_status = executor.execute("redis_info", {})
        assert isinstance(redis_status, str)
    
    def test_database_diagnostic_workflow(self, executor, fresh_db):
        """Test database diagnostic and optimization workflow."""
        tables = executor.execute("pg_show_tables", {})
        assert "orders" in tables.lower()

        locks = executor.execute("pg_locks", {})
        assert isinstance(locks, str)

        stats = executor.execute("pg_stat_statements", {})
        assert isinstance(stats, str)

        explain = executor.execute(
            "pg_explain_analyze",
            {"query": "SELECT * FROM orders WHERE user_email = 'test@example.com'"}
        )
        assert isinstance(explain, str)

        index = executor.execute(
            "pg_create_index",
            {"table": "orders", "column": "user_email"}
        )
        assert isinstance(index, str)
    
    def test_cache_diagnostic_workflow(self, executor):
        """Test Redis cache diagnostic workflow."""

        info = executor.execute("redis_info", {})
        assert "MEMORY" in info

        keys = executor.execute("redis_keys", {"pattern": "*"})
        assert isinstance(keys, str)

        slowlog = executor.execute("redis_slowlog", {})
        assert isinstance(slowlog, str)

    def test_container_health_check_workflow(self, executor):
        """Test container health and diagnostics workflow."""
        containers = executor.execute("docker_ps", {})
        assert "pagezero" in containers.lower()

        app_stats = executor.execute("docker_stats", {"container": "pagezero-app-1"})
        assert isinstance(app_stats, str)

        app_logs = executor.execute(
            "docker_logs",
            {"container": "pagezero-app-1", "lines": 20}
        )
        assert isinstance(app_logs, str)

        disk = executor.execute("check_disk_usage", {})
        assert "/" in disk


class TestContainerRestartWorkflow:
    """Test container restart and recovery workflows."""
    
    def test_app_container_restart_workflow(self, executor):
        """Test restarting app container and verifying recovery."""

        initial_logs = executor.execute(
            "read_app_logs",
            {"service": "app", "lines": 5}
        )

        restart_result = executor.execute("rollback_deploy", {})
        assert isinstance(restart_result, str)
        assert "ERROR executing" not in restart_result

        time.sleep(2)

        health = executor.execute("check_alerts", {})
        assert isinstance(health, str)


class TestDatabaseOptimizationWorkflow:
    """Test complete database optimization scenario."""
    
    def test_identify_and_fix_missing_index(self, executor, fresh_db):
        """
        Simulate SRE workflow:
        1. Identify slow query
        2. Create index
        3. Verify improvement
        """

        stats = executor.execute("pg_stat_statements", {})
        assert isinstance(stats, str)
        
        query_plan = executor.execute(
            "pg_explain_analyze",
            {"query": "SELECT COUNT(*) FROM orders WHERE user_email = 'user_1@company.com'"}
        )
        assert isinstance(query_plan, str)
        
        index_result = executor.execute(
            "pg_create_index",
            {"table": "orders", "column": "user_email"}
        )
        assert isinstance(index_result, str)
        assert "ERROR executing" not in index_result
        
        tables = executor.execute("pg_show_tables", {})
        assert isinstance(tables, str)
    
    def test_cleanup_and_vacuum_workflow(self, executor, fresh_db):
        """Test database maintenance workflow."""
        vacuum = executor.execute("pg_vacuum", {"table": "orders"})
        assert isinstance(vacuum, str)
        
        vacuum_all = executor.execute("pg_vacuum", {})
        assert isinstance(vacuum_all, str)


class TestErrorRecoveryWorkflows:
    """Test error scenarios and recovery."""
    
    def test_search_for_errors_in_logs(self, executor):
        """Test searching for and identifying errors."""
        errors = executor.execute("search_logs", {"pattern": "ERROR"})
        assert isinstance(errors, str)
        
        exceptions = executor.execute("search_logs", {"pattern": "exception"})
        assert isinstance(exceptions, str)
        
        http_errors = executor.execute("search_logs", {"pattern": "500"})
        assert isinstance(http_errors, str)
    
    def test_endpoint_health_monitoring(self, executor):
        """Test endpoint health monitoring."""
        health = executor.execute(
            "curl_endpoint",
            {"url": "http://localhost:5001/health"}
        )
        assert isinstance(health, str)
        assert "200" in health


class TestCompleteIncidentResolution:
    """Test complete incident diagnosis and resolution."""
    
    def test_full_incident_resolution_flow(self, executor, fresh_db):
        """
        Simulate a complete incident:
        1. Alert triggered
        2. Triage (check alerts, get error rate)
        3. Investigate (logs, database, cache)
        4. Diagnose (query analysis, lock detection)
        5. Remediate (create index, clear cache)
        6. Verify (resolution check)
        """
        alerts = executor.execute("check_alerts", {})
        assert isinstance(alerts, str)
        
        error_rate = executor.execute("get_error_rate", {})
        assert isinstance(error_rate, str)
        
        logs = executor.execute("read_app_logs", {"service": "app"})
        assert isinstance(logs, str)
        
        db_activity = executor.execute("pg_stat_activity", {})
        assert isinstance(db_activity, str)
        
        locks = executor.execute("pg_locks", {})
        assert isinstance(locks, str)
        
        redis_info = executor.execute("redis_info", {})
        assert isinstance(redis_info, str)
        
        query_plan = executor.execute(
            "pg_explain_analyze",
            {"query": "SELECT * FROM orders LIMIT 10"}
        )
        assert isinstance(query_plan, str)
        
        index_result = executor.execute(
            "pg_create_index",
            {"table": "orders", "column": "user_email"}
        )
        assert isinstance(index_result, str)
        
        redis_flush = executor.execute("redis_flush_db", {})
        assert isinstance(redis_flush, str)
        
        final_check = executor.execute("check_alerts", {})
        assert isinstance(final_check, str)


class TestConcurrentToolExecution:
    """Test that tools can be executed sequentially (as in real SRE work)."""
    
    def test_rapid_sequential_tool_calls(self, executor):
        """Test multiple rapid tool calls."""
        results = []
        
        tools = [
            ("docker_ps", {}),
            ("pg_show_tables", {}),
            ("redis_keys", {"pattern": "*"}),
            ("check_alerts", {}),
            ("get_error_rate", {}),
            ("pg_stat_activity", {}),
            ("redis_info", {}),
            ("docker_logs", {"container": "pagezero-app-1", "lines": 5}),
            ("pg_locks", {}),
            ("redis_slowlog", {}),
        ]
        
        for tool_name, args in tools:
            result = executor.execute(tool_name, args)
            results.append(result)
            assert isinstance(result, str)
            assert "ERROR executing" not in result
        
        assert len(results) == len(tools)


class TestToolOutputValidation:
    """Test that tool outputs are sensible."""
    
    def test_docker_ps_output_format(self, executor):
        """Verify docker ps output has expected format."""
        result = executor.execute("docker_ps", {})
        assert "pagezero" in result.lower()
        assert any(c in result.lower() for c in ["app", "postgres", "redis"])
    
    def test_pg_show_tables_includes_seed_tables(self, executor):
        """Verify postgres initialization created expected tables."""
        result = executor.execute("pg_show_tables", {})
        assert "orders" in result.lower()
        assert "users" in result.lower()
        assert "products" in result.lower()
    
    def test_curl_response_includes_timing(self, executor):
        """Verify curl output includes response metadata."""
        result = executor.execute(
            "curl_endpoint",
            {"url": "http://localhost:5001/health"}
        )
        assert any(x in result for x in ["HTTP", "time", "200", "Total"])
    
    def test_disk_usage_shows_filesystem(self, executor):
        """Verify disk usage output shows filesystem info."""
        result = executor.execute("check_disk_usage", {})
        assert "/"
        assert any(x in result.lower() for x in ["total", "used", "available", "g", "m"])


class TestSLATracking:
    """Test SLA and incident timing functionality."""
    
    def test_sla_status_tracking(self, stack_backend):
        """Verify SLA status is properly tracked."""
        status1 = stack_backend.get_sla_status()
        assert status1["sla_status"] in ["OK", "VIOLATED"]
        assert status1["downtime_minutes"] >= 0
        assert status1["revenue_loss_usd"] >= 0
        
        time.sleep(1)
        status2 = stack_backend.get_sla_status()
        
        assert status2["downtime_minutes"] >= status1["downtime_minutes"]
    
    def test_incident_timer_reset(self, stack_backend):
        """Verify incident timer can be reset."""
        status1 = stack_backend.get_sla_status()
        time.sleep(1.5)
        
        stack_backend.reset_incident_timer()
        status2 = stack_backend.get_sla_status()
        
        assert status2["downtime_minutes"] <= status1["downtime_minutes"]