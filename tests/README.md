# PageZero-RL Integration Test Suite

Comprehensive integration tests for all PageZero-RL components. **All tests run against REAL containers.**

## Test Coverage

The test suite verifies that ALL functions are callable and that all Docker commands work correctly.

### Test Files

#### `conftest.py` - Pytest Configuration & Fixtures
Shared fixtures and session setup:
- **`containers_running`**: Session-scoped fixture that starts/stops Docker containers
- **`stack_backend`**: Fresh StackBackend instance for each test
- **`executor`**: Fresh Executor instance for each test
- **`cleanup_after_test`**: Auto-cleanup of PostgreSQL and Redis after each test
- **`fresh_db`**: Ensures schema is in clean state before test

#### `test_stack_backend.py` - StackBackend Integration Tests (37 tests)

Tests for all real docker exec commands:

**Docker Commands:**
- `test_docker_ps` - Container listing
- `test_docker_stats` - Container metrics (CPU, memory)
- `test_docker_logs` - Container log retrieval
- `test_docker_restart` - Container restart
- `test_check_disk_usage` - Disk usage info

**PostgreSQL Commands:**
- `test_pg_show_tables` - List tables
- `test_pg_stat_activity` - Active queries
- `test_pg_locks` - Lock detection
- `test_pg_explain_analyze` - Query plans
- `test_pg_stat_statements` - Query statistics
- `test_pg_cancel_query` - Query cancellation
- `test_pg_create_index` - Index creation
- `test_pg_vacuum` - Table maintenance
- SQL injection prevention tests

**Redis Commands:**
- `test_redis_info` - Server stats
- `test_redis_slowlog` - Slow query log
- `test_redis_keys` - Key retrieval
- `test_redis_get_key` - Get key value and TTL
- `test_redis_flush_db` - Database flushing

**Utility Functions:**
- Container reset and cleanup
- Schema drift reversal
- SLA tracking
- Resolution verification

**Edge Cases:**
- Invalid container names
- Malformed SQL
- Shell command errors
- Missing Redis keys

#### `test_executor.py` - Executor Tool Tests (47 tests for 28 tools)

Tests that ALL 28 tools route correctly and execute:

**Layer 1 - Monitoring & Triage:**
- `check_alerts` ✓
- `get_service_metrics` ✓
- `get_error_rate` ✓

**Layer 2 - Application:**
- `read_app_logs` ✓
- `search_logs` ✓
- `get_recent_deploys` ✓
- `rollback_deploy` ✓
- `curl_endpoint` ✓

**Layer 3 - PostgreSQL:**
- `pg_stat_activity` ✓
- `pg_locks` ✓
- `pg_explain_analyze` ✓
- `pg_stat_statements` ✓
- `pg_cancel_query` ✓
- `pg_create_index` ✓
- `pg_vacuum` ✓
- `pg_show_tables` ✓

**Layer 4 - Redis:**
- `redis_info` ✓
- `redis_slowlog` ✓
- `redis_keys` ✓
- `redis_get_key` ✓
- `redis_flush_db` ✓

**Layer 5 - Infrastructure:**
- `docker_ps` ✓
- `docker_stats` ✓
- `docker_restart` ✓
- `docker_logs` ✓
- `check_disk_usage` ✓

**Meta:**
- `diagnose_root_cause` ✓
- `done` ✓

**Error Handling:**
- Missing required arguments
- Invalid arguments
- Unknown tools
- Extra arguments

**Parametrized Test:**
- `test_all_tools_are_callable` - Tests ALL 28 tools in one parametrized test

#### `test_integration.py` - End-to-End Integration Tests (17 tests)

Complete workflows combining multiple tools:

**SRE Workflows:**
- Full investigation workflow (alert → triage → investigate → diagnose)
- Database diagnostic and optimization
- Cache diagnostic and verification
- Container health checks

**Restart Workflows:**
- App container restart and recovery
- Health verification after restart

**Database Optimization:**
- Identify slow queries
- Create indexes
- Vacuum and maintain

**Error Recovery:**
- Search logs for errors
- Endpoint health monitoring
- Error classification

**Complete Incident Resolution:**
- Full incident lifecycle simulation
- Triage → Investigation → Diagnosis → Remediation → Verification

**Load Testing:**
- Rapid sequential tool execution
- Output validation

**SLA Tracking:**
- Downtime tracking
- Revenue loss calculation
- Timer reset

## Quick Start

### Prerequisites

1. **Docker & Docker Compose**: Required to run containers
2. **Python 3.10+**: For running tests
3. **Virtual Environment**: Recommended

### Installation

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
# or
pip install pytest pytest-cov
```

### Running Tests

**Run all tests:**
```bash
pytest 
```

**Run specific test file:**
```bash
pytest tests/test_stack_backend.py
pytest tests/test_executor.py
pytest tests/test_integration.py
```

**Run specific test:**
```bash
pytest tests/test_executor.py::TestExecutorMonitoringTools::test_check_alerts -v
```

**Run all tests with coverage:**
```bash
pytest tests/ --cov=server --cov=tests --cov-report=html
```

**Run tests with verbose output:**
```bash
pytest -v
```

**Run only StackBackend tests:**
```bash
pytest tests/test_stack_backend.py -v
```

**Run only Executor tests:**
```bash
pytest tests/test_executor.py -v
```

**Run only integration tests:**
```bash
pytest tests/test_integration.py -v
```

**Run with detailed failure info:**
```bash
pytest -vv --tb=long
```

## Test Execution Flow

### Session Setup (runs once)
1. `containers_running` fixture starts Docker Compose services
2. Waits for app, postgres, and redis health checks
3. Fixtures remain available for all tests

### Per-Test Setup
1. Fresh `StackBackend` instance created
2. Fresh `Executor` instance created
3. `fresh_db` reverts any schema drift

### Test Execution
- Tests call real docker exec commands
- No mocking - actual database queries execute
- Real Redis commands run
- Real Docker commands execute

### Per-Test Cleanup
1. `cleanup_after_test` terminates stray database connections
2. Redis is flushed
3. State is cleared for next test

### Session Teardown
- Containers remain running (or can be stopped if needed)
