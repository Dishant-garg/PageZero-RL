"""
Pytest configuration and fixtures for PageZero integration tests.
All tests run against REAL containers - no mocks.
"""

import pytest
import subprocess
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from server.stack_backend import StackBackend
from server.executor import Executor


@pytest.fixture(scope="session")
def docker_compose_file():
    """Return path to docker-compose.yml."""
    return str(Path(__file__).parent.parent / "docker-compose.yml")


@pytest.fixture(scope="session")
def containers_running(docker_compose_file):
    """
    Session fixture: ensure containers are up and healthy.
    Just checks that containers exist and are running.
    """
    print("\n🐳 Checking Docker containers...")
    
    max_retries = 20
    for i in range(max_retries):
        try:
            ps_result = subprocess.run(
                f"docker ps --filter 'name=pagezero' --format '{{{{.Names}}}}'",
                shell=True,
                capture_output=True,
                text=True,
                timeout=5
            )
            
            containers = ps_result.stdout.strip().split('\n')
            container_names = [c for c in containers if c.strip()]
            
            if len(container_names) >= 3:
                print(f"✅ All containers running: {', '.join(container_names)}")
                break
            
        except Exception as e:
            pass
        
        if i == 0:
            print("📦 Starting Docker containers...")
            result = subprocess.run(
                f"docker compose -f {docker_compose_file} up -d",
                shell=True,
                capture_output=True,
                text=True,
                timeout=60
            )
        
        if i == max_retries - 1:
            raise RuntimeError(f"Containers failed to start after {max_retries} retries")
        time.sleep(1)
    
    yield
    
    print("✨ Containers ready for testing")


@pytest.fixture
def stack_backend(containers_running):
    """
    Function fixture: fresh StackBackend instance for each test.
    Ensures containers are up before test runs.
    """
    backend = StackBackend()
    backend.reset_incident_timer()
    yield backend


@pytest.fixture
def executor(stack_backend):
    """
    Function fixture: fresh Executor instance for each test.
    Uses the stack_backend fixture.
    """
    return Executor(stack_backend)


@pytest.fixture
def cleanup_after_test(stack_backend):
    """
    Fixture to clean up PostgreSQL and Redis after each test.
    Ensures no state leaks between tests.
    """
    yield
    try:
        stack_backend.cleanup_postgres()
        stack_backend.cleanup_redis()
    except Exception:
        pass


@pytest.fixture
def fresh_db(stack_backend):
    """
    Fixture that resets schema drift before test.
    Ensures test starts with known schema state.
    """
    stack_backend.revert_schema_drift()
    return stack_backend
