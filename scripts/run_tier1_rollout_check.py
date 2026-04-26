#!/usr/bin/env python3
"""Run the same 30-episode mocked rollout check as ``tests/test_tier1_rollout.py``.

Usage (from repo root, no GPU / no Docker):

  python scripts/run_tier1_rollout_check.py

Exit 0 only if median env steps per episode >= 3.
"""

from __future__ import annotations

import os
import statistics
import sys

# Repo root on sys.path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models import PageZeroAction  # noqa: E402


def main() -> int:
    from tests.test_tier1_rollout import _minimal_env_for_step_tests

    counts: list[int] = []
    for _ in range(30):
        env = _minimal_env_for_step_tests()
        env._step_count = 0
        env._history = []
        env._call_counts = {}
        for tool in ("check_alerts", "check_disk_usage", "docker_ps"):
            env.step(PageZeroAction(tool=tool, args={}))
        counts.append(env._step_count)

    med = float(statistics.median(counts))
    print(f"30 episodes × 3 tools → step counts: all == 3 ? {all(c == 3 for c in counts)}")
    print(f"median num_steps: {med}")
    if med < 3.0:
        print("FAIL: median < 3", file=sys.stderr)
        return 1
    print("PASS: median >= 3 (server allows multi-step rollouts under Tier-1 mocks).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
