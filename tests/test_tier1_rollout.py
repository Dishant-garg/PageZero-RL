"""
Tier 1 rollout checks: early documentation guard + multi-step capacity.

The 30-episode / median >= 3 check uses a minimal mocked PageZeroEnvironment
(no Docker) to prove the server allows ≥3 steps per episode when tools chain.
Full GRPO median is validated separately on GPU runs via trajectories.jsonl.
"""

from __future__ import annotations

import statistics
from unittest.mock import MagicMock

import pytest

from models import PageZeroAction


def _minimal_env_for_step_tests():
    """Build PageZeroEnvironment without running __init__ (no Docker)."""
    from server.PageZero_environment import PageZeroEnvironment

    env = PageZeroEnvironment.__new__(PageZeroEnvironment)
    env.backend = MagicMock()
    env.backend.get_sla_status.return_value = {
        "sla_status": "OK",
        "revenue_loss_usd": 0.0,
        "downtime_minutes": 0.0,
    }
    env.backend.verify_resolution.return_value = False
    env.backend.cleanup_postgres = MagicMock()
    env.backend.cleanup_redis = MagicMock()
    env.backend.revert_schema_drift = MagicMock()
    env.backend.reset_incident_timer = MagicMock()
    env.executor = MagicMock()
    env.executor.execute.return_value = "MOCK_OK"
    env.judge = MagicMock()
    env.judge.get_phase_reward.return_value = (0.1, "mock judge")
    env.judge.verify_resolution.return_value = (False, "mock")
    env.judge.evaluate_terminal.return_value = (0.0, 0.0, "mock")
    env.judge._canonical_score = MagicMock(return_value=0.0)
    env.curriculum = MagicMock()
    env.curriculum.record_result = MagicMock()
    env.drift_engine = MagicMock()
    env.drift_engine.maybe_drift.return_value = None
    env._scenario = {
        "alert": "test alert",
        "name": "test scenario",
        "task_id": "task_1",
        "layer": "postgres",
        "inject_commands": [],
    }
    env._max_steps = 15
    env._episode_count = 0
    env._cumulative_reward = 0.0
    env._call_counts = {}
    env._used_diagnose_root_cause = False
    env._used_write_postmortem = False
    env._state = MagicMock()
    env._state.difficulty = 0.5
    env._state.is_resolved = False
    env._state.cumulative_reward = 0.0
    env._snapshot_text = MagicMock(return_value="snapshot")
    return env


def test_diagnose_root_cause_blocked_on_steps_1_and_2():
    from server.config import REWARD_EARLY_DOC_BLOCK

    env = _minimal_env_for_step_tests()
    for attempt in (1, 2):
        env._step_count = 0
        env._history = []
        env._call_counts = {}
        obs = env.step(PageZeroAction(tool="diagnose_root_cause", args={"root_cause": "x"}))
        assert obs.step == attempt
        assert "BLOCKED" in (obs.tool_output or "")
        assert obs.reward == pytest.approx(float(REWARD_EARLY_DOC_BLOCK))
        env.judge.get_phase_reward.assert_not_called()


def test_diagnose_root_cause_allowed_from_step_3():
    env = _minimal_env_for_step_tests()
    env._step_count = 0
    env._history = []
    env._call_counts = {}
    env.step(PageZeroAction(tool="check_alerts", args={}))
    env.step(PageZeroAction(tool="check_disk_usage", args={}))
    env.judge.get_phase_reward.reset_mock()
    obs = env.step(PageZeroAction(tool="diagnose_root_cause", args={"root_cause": "real"}))
    assert obs.step == 3
    env.executor.execute.assert_called()
    env.judge.get_phase_reward.assert_called()


def test_thirty_episodes_median_steps_at_least_three():
    counts: list[int] = []
    for _ in range(30):
        env = _minimal_env_for_step_tests()
        env._step_count = 0
        env._history = []
        env._call_counts = {}
        # Three distinct non-done tools → 3 env steps, no repeat breaker.
        for tool in ("check_alerts", "check_disk_usage", "docker_ps"):
            env.step(PageZeroAction(tool=tool, args={}))
        counts.append(env._step_count)

    med = float(statistics.median(counts))
    assert med >= 3.0, f"median steps {med}, counts sample={counts[:5]}..."
    assert all(c == 3 for c in counts)


def test_train_py_tier1_defaults_in_source():
    """train.py defaults reflect Tier 1 without importing trl/torch."""
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "train.py").read_text()
    assert '--max-new-tokens", type=int, default=1024' in text
    assert '--temperature", type=float, default=0.6' in text
    assert '--top-p", type=float, default=0.9' in text
    assert "mask_truncated_completions=False" in text


def test_server_has_min_steps_before_resolve_guard_in_source():
    from pathlib import Path

    env_text = (Path(__file__).resolve().parents[1] / "server" / "PageZero_environment.py").read_text()
    cfg_text = (Path(__file__).resolve().parents[1] / "server" / "config.py").read_text()
    assert "MIN_STEPS_BEFORE_RESOLVE" in cfg_text
    assert "gate_resolve_ready = (" in env_text
    assert "gate_resolved and self._step_count >= int(MIN_STEPS_BEFORE_RESOLVE)" in env_text
    assert "Resolution confirmed but deferred until step >= " in env_text


def test_server_has_docs_done_gate_constants_in_source():
    from pathlib import Path

    env_text = (Path(__file__).resolve().parents[1] / "server" / "PageZero_environment.py").read_text()
    cfg_text = (Path(__file__).resolve().parents[1] / "server" / "config.py").read_text()
    assert "REQUIRE_DOCS_BEFORE_SUCCESS" in cfg_text
    assert "REWARD_DONE_UNRESOLVED" in cfg_text
    assert "REWARD_DONE_BEFORE_DOCS" in cfg_text
    assert "docs_complete = (" in env_text
    assert "docs_missing_done = (done_step_ready and gate_resolve_ready and not docs_ready)" in env_text


def test_done_unresolved_is_ignored_with_penalty_feedback():
    from server.config import REWARD_DONE_UNRESOLVED

    env = _minimal_env_for_step_tests()
    env._step_count = 4  # next step is 5; past min-done floor
    env._history = []
    env._call_counts = {}
    env.backend.verify_resolution.return_value = False
    obs = env.step(PageZeroAction(tool="done", args={}))
    assert obs.is_done is False
    assert "stack not resolved yet" in (obs.judge_feedback or "")
    assert obs.reward <= float(REWARD_DONE_UNRESOLVED) + 0.2


def test_done_resolved_but_docs_missing_is_deferred():
    from server.config import REWARD_DONE_BEFORE_DOCS

    env = _minimal_env_for_step_tests()
    env._step_count = 4  # next step is 5; past min floors
    env._history = []
    env._call_counts = {}
    env.backend.verify_resolution.return_value = True
    env.judge.verify_resolution.return_value = (True, "ok")
    env._used_diagnose_root_cause = False
    env._used_write_postmortem = False
    obs = env.step(PageZeroAction(tool="done", args={}))
    assert obs.is_done is False
    assert "documentation incomplete" in (obs.judge_feedback or "")
    assert obs.reward <= float(REWARD_DONE_BEFORE_DOCS) + 0.2


def test_done_resolved_with_docs_can_terminate():
    env = _minimal_env_for_step_tests()
    env._step_count = 4  # next step is 5; past min floors
    env._history = []
    env._call_counts = {}
    env.backend.verify_resolution.return_value = True
    env.judge.verify_resolution.return_value = (True, "ok")
    env._used_diagnose_root_cause = True
    env._used_write_postmortem = True
    obs = env.step(PageZeroAction(tool="done", args={}))
    assert obs.is_done is True
