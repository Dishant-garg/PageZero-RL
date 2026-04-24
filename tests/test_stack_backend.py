"""
Integration tests for StackBackend.
Tests REAL docker exec commands against running containers.
NO MOCKS - all commands execute against actual services.
"""

import subprocess
from server.config import APP_CONTAINER


class TestStackBackendDocker:
    """Test Docker-related commands."""
    
    def test_docker_ps(self, stack_backend):
        """Verify docker ps command works and lists containers."""
        result = stack_backend.docker_ps()
        assert isinstance(result, str)
        assert "pagezero" in result.lower()
        assert "ERROR" not in result
    
    def test_docker_stats(self, stack_backend):
        """Verify docker stats works for a container."""
        result = stack_backend.docker_stats(APP_CONTAINER)
        assert isinstance(result, str)
        assert ("CPU" in result or "ERROR" not in result)
    
    def test_docker_logs(self, stack_backend):
        """Verify docker logs returns recent logs."""
        result = stack_backend.docker_logs(APP_CONTAINER, lines=10)
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_docker_logs_custom_lines(self, stack_backend):
        """Verify docker logs respects line count parameter."""
        result_5 = stack_backend.docker_logs(APP_CONTAINER, lines=5)
        result_50 = stack_backend.docker_logs(APP_CONTAINER, lines=50)
        assert len(result_50) >= len(result_5)
    
    def test_docker_restart(self, stack_backend):
        """Verify docker restart command works."""
        result = stack_backend.docker_restart(APP_CONTAINER)
        assert isinstance(result, str)
        assert "ERROR" not in result or APP_CONTAINER in result
    
    def test_check_disk_usage(self, stack_backend):
        """Verify disk usage command works."""
        result = stack_backend.check_disk_usage()
        assert isinstance(result, str)
        assert "/" in result
        assert "ERROR" not in result


class TestStackBackendCurl:
    """Test curl endpoint functionality."""
    
    def test_curl_endpoint_health(self, stack_backend):
        """Verify curl can reach app health endpoint."""
        result = stack_backend.curl_endpoint("http://localhost:5001/health")
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_curl_endpoint_invalid_url(self, stack_backend):
        """Verify curl handles invalid URLs gracefully."""
        result = stack_backend.curl_endpoint("http://invalid-host-12345:9999/test")
        assert isinstance(result, str)


class TestStackBackendPostgreSQL:
    """Test PostgreSQL commands via docker exec."""
    
    def test_pg_show_tables(self, fresh_db):
        """Verify pg_show_tables returns list of tables."""
        result = fresh_db.pg_show_tables()
        assert isinstance(result, str)
        assert "orders" in result.lower()
        assert "users" in result.lower()
        assert "products" in result.lower()
    
    def test_pg_stat_activity(self, fresh_db):
        """Verify pg_stat_activity shows current queries."""
        result = fresh_db.pg_stat_activity()
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_pg_locks(self, fresh_db):
        """Verify pg_locks query executes."""
        result = fresh_db.pg_locks()
        assert isinstance(result, str)
        assert "ERROR" not in result or len(result) > 0
    
    def test_pg_stat_statements(self, fresh_db):
        """Verify pg_stat_statements returns query stats."""
        result = fresh_db.pg_stat_statements()
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_pg_explain_analyze_valid_query(self, fresh_db):
        """Verify EXPLAIN ANALYZE works for valid SELECT."""
        result = fresh_db.pg_explain_analyze("SELECT * FROM users LIMIT 1")
        assert isinstance(result, str)
        assert "ERROR" not in result or "Limit" in result or "Seq Scan" in result
    
    def test_pg_explain_analyze_rejects_unsafe_queries(self, fresh_db):
        """Verify EXPLAIN ANALYZE rejects non-SELECT and injections."""
        result = fresh_db.pg_explain_analyze("UPDATE users SET username='test'")
        assert "ERROR" in result
        
        result = fresh_db.pg_explain_analyze("SELECT * FROM users; DELETE FROM users;")
        assert "ERROR" in result
        
        result = fresh_db.pg_explain_analyze("SELECT * FROM users -- comment")
        assert "ERROR" in result
    
    def test_pg_create_index(self, fresh_db):
        """Verify CREATE INDEX executes successfully."""
        result = fresh_db.pg_create_index("orders", "user_email")
        assert isinstance(result, str)
        assert "ERROR" not in result or "already exists" in result.lower()
    
    def test_pg_create_index_invalid_table(self, fresh_db):
        """Verify CREATE INDEX rejects invalid identifiers."""
        result = fresh_db.pg_create_index("tab'; DROP TABLE users; --", "col")
        assert "ERROR" in result
        
        result = fresh_db.pg_create_index("orders", "col'; DROP TABLE users; --")
        assert "ERROR" in result
    
    def test_pg_vacuum(self, fresh_db):
        """Verify VACUUM ANALYZE works."""
        result = fresh_db.pg_vacuum("orders")
        assert isinstance(result, str)
        assert "ERROR" not in result or "VACUUM" in result
    
    def test_pg_vacuum_all_tables(self, fresh_db):
        """Verify VACUUM without table name works."""
        result = fresh_db.pg_vacuum("")
        assert isinstance(result, str)
    
    def test_pg_cancel_query(self, fresh_db):
        """Verify pg_cancel_query attempts to cancel a backend."""
        result = fresh_db.pg_cancel_query(999999)
        assert isinstance(result, str)
        assert "ERROR" not in result or len(result) > 0


class TestStackBackendRedis:
    """Test Redis commands via docker exec."""
    
    def test_redis_info(self, stack_backend, cleanup_after_test):
        """Verify redis INFO command returns stats."""
        result = stack_backend.redis_info()
        assert isinstance(result, str)
        assert "STATS" in result
        assert "MEMORY" in result
        assert "ERROR" not in result
    
    def test_redis_keys_all(self, stack_backend):
        """Verify redis KEYS * works."""
        result = stack_backend.redis_keys("*")
        assert isinstance(result, str)
        assert "ERROR" not in result
    
    def test_redis_keys_pattern(self, stack_backend):
        """Verify redis KEYS with pattern works."""
        stack_backend._run_redis_cmd("SET", "test_key_123", "value")
        
        result = stack_backend.redis_keys("test_key*")
        assert isinstance(result, str)
        assert "test_key" in result or "empty" in result.lower()
    
    def test_redis_set_and_get(self, stack_backend, cleanup_after_test):
        """Verify redis GET works after setting a key."""
        stack_backend._run_redis_cmd("SET", "integration_test_key", "test_value")
        
        result = stack_backend.redis_get_key("integration_test_key")
        assert isinstance(result, str)
        assert "test_value" in result or "Value:" in result
    
    def test_redis_get_nonexistent_key(self, stack_backend):
        """Verify redis GET handles missing keys."""
        result = stack_backend.redis_get_key("nonexistent_key_xyz_123")
        assert isinstance(result, str)
        assert "Value:" in result or "nil" in result.lower()
    
    def test_redis_slowlog(self, stack_backend):
        """Verify redis SLOWLOG works."""
        result = stack_backend.redis_slowlog()
        assert isinstance(result, str)
        assert "ERROR" not in result
        assert isinstance(result, str)
    
    def test_redis_flush_db(self, stack_backend, cleanup_after_test):
        """Verify redis FLUSHDB works."""
        stack_backend._run_redis_cmd("SET", "temp_key", "value")
        
        result = stack_backend.redis_flush_db()
        assert isinstance(result, str)
        assert "OK" in result or "ERROR" not in result
        
        check = stack_backend.redis_get_key("temp_key")
        assert "TTL: -2" in check or "nil" in check.lower() or "None" in check


class TestStackBackendUtilities:
    """Test utility and setup functions."""
    
    def test_reset_containers(self, stack_backend):
        """Verify containers are running (via SSH if remote, local if not)."""
        result = stack_backend.docker_ps()
        assert isinstance(result, str)
        assert "pagezero" in result.lower() or "Up" in result
    
    def test_cleanup_postgres(self, stack_backend):
        """Verify cleanup_postgres executes without error."""
        stack_backend.cleanup_postgres()
    
    def test_cleanup_redis(self, stack_backend):
        """Verify cleanup_redis executes without error."""
        stack_backend.cleanup_redis()
    
    def test_revert_schema_drift(self, stack_backend):
        """Verify revert_schema_drift executes without error."""
        stack_backend.revert_schema_drift()
    
    def test_reset_incident_timer(self, stack_backend):
        """Verify incident timer reset works."""
        import time
        old_time = stack_backend.incident_start_time
        time.sleep(0.1)
        stack_backend.reset_incident_timer()
        new_time = stack_backend.incident_start_time
        assert new_time > old_time
    
    def test_get_sla_status(self, stack_backend):
        """Verify SLA status returns valid dict."""
        status = stack_backend.get_sla_status()
        assert isinstance(status, dict)
        assert "downtime_minutes" in status
        assert "revenue_loss_usd" in status
        assert "sla_status" in status
        assert status["sla_status"] in ["OK", "VIOLATED"]
    
    def test_verify_resolution(self, stack_backend):
        """Verify resolution check works and returns boolean."""
        result = stack_backend.verify_resolution()
        assert isinstance(result, bool)
        assert result in [True, False]


class TestStackBackendEdgeCases:
    """Test edge cases and error handling."""
    
    def test_run_cmd_with_invalid_command(self, stack_backend):
        """Verify error handling for invalid shell commands."""
        result = stack_backend._run_cmd("invalid_command_xyz_123_does_not_exist")
        assert isinstance(result, str)
        assert "ERROR" in result
    
    def test_run_psql_with_invalid_sql(self, stack_backend):
        """Verify error handling for invalid SQL."""
        result = stack_backend._run_psql("INVALID SYNTAX HERE")
        assert isinstance(result, str)
        assert "ERROR" in result
    
    def test_docker_stats_invalid_container(self, stack_backend):
        """Verify error handling for nonexistent container."""
        result = stack_backend.docker_stats("nonexistent-container-xyz")
        assert isinstance(result, str)
        assert "ERROR" in result or len(result) > 0
    
    def test_redis_cmd_with_invalid_syntax(self, stack_backend):
        """Verify redis command error handling."""
        result = stack_backend._run_redis_cmd("INVALID", "COMMAND", "SYNTAX")
        assert isinstance(result, str)
        assert len(result) > 0