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
    Handles both local and remote (SSH) environments via StackBackend.
    """
    from server.stack_backend import VM_HOST, StackBackend
    
    # Create a temporary backend just for health checking
    backend = StackBackend()
    
    print(f"\n🐳 Checking Docker containers on {VM_HOST}...")
    
    max_retries = 20
    for i in range(max_retries):
        try:
            # Use backend's docker_ps() which handles SSH automatically
            ps_output = backend.docker_ps()
            
            # Count running containers - looking for 'pagezero' and 'Up'
            running_count = sum(1 for line in ps_output.split('\n') if "pagezero" in line and ("Up" in line or "running" in line))
            
            if running_count >= 3:
                print(f"✅ All containers running on {VM_HOST} ({running_count} found)")
                break
            
            # Only attempt to start containers if running on localhost
            if i == 0 and VM_HOST == "localhost":
                print("📦 Starting local Docker containers...")
                subprocess.run(
                    f"docker compose -f {docker_compose_file} up -d",
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
            elif i == 0:
                print(f"⚠️  Remote containers not fully up on {VM_HOST} ({running_count}/3). Waiting...")

        except Exception as e:
            if i == 0:
                print(f"❌ Error checking containers: {e}")
        
        if i == max_retries - 1:
            raise RuntimeError(f"Containers failed to start/be-found on {VM_HOST} after {max_retries} retries. Output: {ps_output if 'ps_output' in locals() else 'None'}")
        time.sleep(2)
    
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
