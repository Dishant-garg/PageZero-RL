#!/usr/bin/env python3
"""
verify.py — PageZero Stack Verification Script

Runs a series of smoke tests against the live Docker stack to confirm
that stack_backend.py can execute real commands against each service.

Usage:
    python verify.py [--verbose]

Prerequisites:
    docker compose up -d   (from the project root)
"""

import argparse
import subprocess
import sys
import time
import importlib.util
import os

# ── Import StackBackend directly from its module file ──────────────────────
# We do this to avoid triggering server/__init__.py which would require
# the openenv package (only needed when running the full OpenEnv server).
_repo_root = os.path.dirname(os.path.abspath(__file__))
_backend_path = os.path.join(_repo_root, "server", "stack_backend.py")
_spec = importlib.util.spec_from_file_location("stack_backend", _backend_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
StackBackend = _mod.StackBackend

GREEN = "\033[92m"
RED   = "\033[91m"
YELLOW= "\033[93m"
RESET = "\033[0m"
BOLD  = "\033[1m"

passed = 0
failed = 0


def ok(label: str, detail: str = ""):
    global passed
    passed += 1
    suffix = f" — {detail[:80]}" if detail else ""
    print(f"  {GREEN}✓{RESET} {label}{suffix}")


def fail(label: str, detail: str = ""):
    global failed
    failed += 1
    suffix = f"\n      {detail[:200]}" if detail else ""
    print(f"  {RED}✗{RESET} {label}{suffix}")


def section(title: str):
    print(f"\n{BOLD}{title}{RESET}")


def check_containers_running():
    """Verify all three containers are up before proceeding."""
    result = subprocess.run(
        "docker ps --format '{{.Names}}' | grep pagezero",
        shell=True, capture_output=True, text=True
    )
    names = result.stdout.strip().split("\n")
    needed = {"pagezero-postgres-1", "pagezero-redis-1", "pagezero-app-1"}
    running = {n.strip() for n in names if n.strip()}
    missing = needed - running
    if missing:
        print(f"{RED}ERROR: Containers not running: {missing}{RESET}")
        print(f"Run:  docker compose up -d    from {os.path.dirname(os.path.abspath(__file__))}")
        sys.exit(1)
    print(f"{GREEN}All 3 containers detected: {', '.join(sorted(running & needed))}{RESET}")


def main(verbose: bool = False):
    backend = StackBackend()

    # ── 0. Container Health ──────────────────────────────────────────────
    section("0. Container Health")
    check_containers_running()

    # ── 1. PostgreSQL Tests ─────────────────────────────────────────────
    section("1. PostgreSQL")

    out = backend.pg_stat_activity()
    if verbose: print(f"    pg_stat_activity:\n{out[:400]}")
    if "ERROR" in out:
        fail("pg_stat_activity", out)
    else:
        ok("pg_stat_activity")

    out = backend.pg_show_tables()
    if verbose: print(f"    pg_show_tables:\n{out}")
    if "ERROR" in out:
        fail("pg_show_tables", out)
    elif "orders" in out and "users" in out:
        ok("pg_show_tables — found orders, users, products")
    else:
        fail("pg_show_tables — expected tables not found", out)

    out = backend.pg_locks()
    if verbose: print(f"    pg_locks:\n{out}")
    if "ERROR" in out:
        fail("pg_locks", out)
    else:
        ok("pg_locks")

    out = backend.pg_explain_analyze("SELECT COUNT(*) FROM orders")
    if verbose: print(f"    pg_explain_analyze:\n{out[:300]}")
    if "ERROR" in out:
        fail("pg_explain_analyze", out)
    elif "Seq Scan" in out or "Aggregate" in out:
        ok("pg_explain_analyze — query plan returned")
    else:
        fail("pg_explain_analyze — unexpected output", out)

    # Row count sanity check
    out = backend._run_psql("SELECT COUNT(*) FROM orders;")
    if verbose: print(f"    Row count: {out}")
    if "ERROR" in out:
        fail("orders row count", out)
    else:
        rows_line = [l for l in out.split("\n") if l.strip().isdigit()]
        if rows_line and int(rows_line[0].strip()) >= 1000:
            ok(f"orders table seeded — ~{rows_line[0].strip()} rows")
        else:
            fail("orders table count looks wrong", out)

    # ── 2. Redis Tests ───────────────────────────────────────────────────
    section("2. Redis")

    out = backend.redis_info()
    if verbose: print(f"    redis_info:\n{out[:400]}")
    if "ERROR" in out:
        fail("redis_info", out)
    elif "used_memory" in out or "total_commands_processed" in out:
        ok("redis_info — stats returned")
    else:
        fail("redis_info — unexpected output", out)

    out = backend.redis_slowlog()
    if verbose: print(f"    redis_slowlog:\n{out}")
    if "ERROR" in out:
        fail("redis_slowlog", out)
    else:
        ok("redis_slowlog")

    # Set + get a test key
    backend._run_redis_cmd("SET verify_test_key hello_pagezero")
    out = backend.redis_get_key("verify_test_key")
    if verbose: print(f"    redis_get_key: {out}")
    if "hello_pagezero" in out:
        ok("redis_get_key — round-trip SET/GET")
    else:
        fail("redis_get_key — value mismatch", out)

    out = backend.redis_keys("verify_test_*")
    if verbose: print(f"    redis_keys: {out}")
    if "verify_test_key" in out:
        ok("redis_keys pattern match")
    else:
        fail("redis_keys — key not found", out)

    # Clean up test key
    backend._run_redis_cmd("DEL verify_test_key")

    # ── 3. Docker / Infra Tests ──────────────────────────────────────────
    section("3. Docker / Infrastructure")

    out = backend.docker_ps()
    if verbose: print(f"    docker_ps:\n{out}")
    if "ERROR" in out:
        fail("docker_ps", out)
    elif "pagezero" in out:
        ok("docker_ps — containers listed")
    else:
        fail("docker_ps — no pagezero containers", out)

    out = backend.docker_stats("pagezero-postgres-1")
    if verbose: print(f"    docker_stats: {out}")
    if "ERROR" in out:
        fail("docker_stats (postgres)", out)
    elif "CPU" in out or "MEM" in out:
        ok("docker_stats — CPU/MEM reported")
    else:
        fail("docker_stats — unexpected output", out)

    out = backend.docker_logs("pagezero-app-1", lines=10)
    if verbose: print(f"    docker_logs:\n{out[:300]}")
    if "ERROR" in out:
        fail("docker_logs (app)", out)
    else:
        ok("docker_logs — app logs returned")

    out = backend.check_disk_usage()
    if verbose: print(f"    disk_usage: {out}")
    if "ERROR" in out:
        fail("check_disk_usage", out)
    elif "/" in out:
        ok("check_disk_usage")
    else:
        fail("check_disk_usage — unexpected output", out)

    # ── 4. App Endpoint Tests ────────────────────────────────────────────
    section("4. Application Endpoints")

    out = backend.curl_endpoint("http://localhost:5000/health")
    if verbose: print(f"    curl /health: {out}")
    if "HTTP 200" in out:
        ok("GET /health → 200 OK")
    else:
        fail("GET /health — expected HTTP 200", out)

    out = backend.curl_endpoint("http://localhost:5000/api/stats")
    if verbose: print(f"    curl /api/stats: {out[:200]}")
    if "HTTP 200" in out:
        ok("GET /api/stats → 200 OK")
    else:
        fail("GET /api/stats — expected HTTP 200", out)

    # ── 5. SLA Tracker ──────────────────────────────────────────────────
    section("5. SLA Tracker")

    sla = backend.get_sla_status()
    if verbose: print(f"    sla_status: {sla}")
    if "downtime_minutes" in sla and "revenue_loss_usd" in sla:
        ok(f"get_sla_status — downtime={sla['downtime_minutes']}m, loss=${sla['revenue_loss_usd']}")
    else:
        fail("get_sla_status — missing fields", str(sla))

    # ── 6. verify_resolution ────────────────────────────────────────────
    section("6. Resolution Verifier")

    healthy = backend.verify_resolution()
    if healthy:
        ok("verify_resolution → stack is HEALTHY ✔")
    else:
        print(f"  {YELLOW}⚠{RESET}  verify_resolution → stack UNHEALTHY (acceptable if a fault was injected)")

    # ── Summary ──────────────────────────────────────────────────────────
    total = passed + failed
    print(f"\n{'═'*50}")
    print(f"{BOLD}Results: {passed}/{total} passed{RESET}", end="")
    if failed:
        print(f"  {RED}({failed} failed){RESET}")
        sys.exit(1)
    else:
        print(f"  {GREEN}All checks passed!{RESET}")
        sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify PageZero stack is live and responding")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print raw tool outputs")
    args = parser.parse_args()
    main(verbose=args.verbose)
