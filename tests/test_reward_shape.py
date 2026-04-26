"""Focused unit tests for the kube-sre-gym-aligned reward shape.

These tests are deliberately *self-contained* — they do not need the Docker
backend, the Gemini API, or the WebSocket env. They cover the pure-Python
logic that previously had no coverage and that we keep accidentally
breaking when we touch the reward stack:

    1. ``MasteryCurriculum`` advancement gates (min_episodes, advance_rate,
       fast-track, weak-spot biased sampling).
    2. ``RewardLogger`` writes a single canonical CSV header + JSONL line per
       episode, even across multiple ``log()`` calls.
    3. ``make_reward_total`` returns ``no_tool_penalty`` when the agent
       produced zero tool calls (env was never stepped) and ``env.total_reward``
       otherwise — and calls ``maybe_log_and_record`` exactly once.
    4. ``reward_diagnosis_metric`` / ``reward_fix_metric`` extract their
       respective buckets without double-counting.
    5. ``PageZeroToolEnv._split_terminal`` cleanly splits the terminal step
       into per-step + estimated-terminal halves.
    6. ``PageZeroToolEnv.trajectory_payload`` reports ``normalized_reward =
       sum/n_steps`` and surfaces ``stack_healthy``.
    7. ``persona_for_difficulty`` maps the curriculum tier to the right judge
       persona (junior < 0.4, senior < 0.7, principal otherwise).
    8. ``phase_score`` + ``get_skipped_phases`` deterministic shaping match
       the kube-sre-gym magnitudes wired in ``server/config.py``.
"""

import csv
import json
import sys
import types
from pathlib import Path

import pytest

# Make the repo root importable without relying on conftest fixtures (which
# spin up Docker containers we don't need for these unit tests).
sys.path.insert(0, str(Path(__file__).parent.parent))


# -----------------------------------------------------------------------------
# Lightweight stubs for the heavy ML deps that ``train`` imports at module
# scope. These tests exercise pure-Python reward / curriculum / logger logic
# only — no model, no tokenizer, no GRPO trainer — so we avoid pulling in
# torch / trl / transformers which (a) we don't need, (b) seg-fault on
# macOS without the full ML extras installed.
# -----------------------------------------------------------------------------

def _install_ml_stubs() -> None:
    if "trl" not in sys.modules:
        trl_mod = types.ModuleType("trl")

        class _Stub:
            def __init__(self, *a, **kw):
                pass

        trl_mod.GRPOConfig = _Stub
        trl_mod.GRPOTrainer = _Stub
        sys.modules["trl"] = trl_mod

    if "transformers" not in sys.modules:
        tf_mod = types.ModuleType("transformers")
        tf_mod.AutoTokenizer = type("AutoTokenizer", (), {})
        sys.modules["transformers"] = tf_mod

    if "datasets" not in sys.modules:
        ds_mod = types.ModuleType("datasets")

        class _StubDataset:
            def __init__(self, data: dict):
                self.column_names = list(data.keys())
                lens = {len(v) for v in data.values()}
                if len(lens) != 1:
                    raise ValueError("ragged columns")
                self._n = lens.pop()
                self._data = data

            def __len__(self) -> int:
                return self._n

            def __getitem__(self, i: int):
                return {k: self._data[k][i] for k in self._data}

        class Dataset:
            @staticmethod
            def from_dict(data: dict):
                return _StubDataset(data)

        ds_mod.Dataset = Dataset
        sys.modules["datasets"] = ds_mod


_install_ml_stubs()

from train import (  # noqa: E402  (import after stub install)
    TASK_ALERT_ONE_LINER,
    MasteryCurriculum,
    PageZeroToolEnv,
    RewardLogger,
    build_grpo_dataset,
    make_reward_total,
    reward_diagnosis_metric,
    reward_fix_metric,
)
from server.config import (
    REWARD_BACKWARD_PHASE,
    REWARD_CORRECT_ORDER,
    REWARD_SKIPPED_PHASE,
    REWARD_TRIAGE_FIRST,
    REWARD_WRONG_FIRST,
)
from server.llm_judge import (
    get_skipped_phases,
    persona_for_difficulty,
    phase_score,
)


# =============================================================================
# Test doubles — minimal stand-ins for a stepped PageZeroToolEnv
# =============================================================================

class _FakeEnv:
    """Stand-in for ``PageZeroToolEnv`` that lets us drive ``reward_total``.

    Only the fields touched by ``make_reward_total`` /
    ``reward_diagnosis_metric`` / ``reward_fix_metric`` are populated.
    """

    def __init__(
        self,
        trajectory=None,
        total_reward=0.0,
        diagnosis_reward=0.0,
        fix_reward=0.0,
    ):
        self.trajectory = list(trajectory or [])
        self.total_reward = float(total_reward)
        self.diagnosis_reward = float(diagnosis_reward)
        self.fix_reward = float(fix_reward)
        self.log_calls = 0

    def maybe_log_and_record(self):
        self.log_calls += 1


# =============================================================================
# 1) MasteryCurriculum
# =============================================================================

class TestMasteryCurriculum:
    def _curriculum(self):
        return MasteryCurriculum(stage_task_pools={
            "easy": ["task_1", "task_2"],
            "medium": ["task_3", "task_4"],
            "hard": ["task_5"],
        })

    def test_starts_at_first_tier(self):
        # Tier ladder is independent of stage ("easy"/"medium"/"hard") names.
        # train.py defines DIFFICULTY_TIERS = [warmup, beginner, intermediate,
        # advanced, expert] — so a fresh curriculum must start at "warmup".
        c = self._curriculum()
        assert c.get_tier_name() == "warmup"

    def test_does_not_advance_below_min_episodes_without_fast_track(self):
        # Warmup needs min_episodes (>=5) AND advance_rate≥0.6, *unless*
        # fast-track triggers (≥3 episodes at 90%+). A single success must
        # not advance the tier.
        c = self._curriculum()
        c.record(task_id="task_1", normalized_reward=1.0, resolved=True)
        assert c.get_tier_name() == "warmup"

    def test_advances_after_sustained_success_streak(self):
        c = self._curriculum()
        start = c.get_tier_name()
        for _ in range(8):
            c.record(task_id="task_1", normalized_reward=0.8, resolved=True)
        assert c.get_tier_name() != start, (
            "MasteryCurriculum failed to advance after a sustained success streak"
        )

    def test_fast_track_advances_after_three_perfect_episodes(self):
        c = self._curriculum()
        start = c.get_tier_name()
        for _ in range(3):
            c.record(task_id="task_1", normalized_reward=1.0, resolved=True)
        assert c.get_tier_name() != start, (
            "Fast-track gate (>=3 episodes at 90%+) failed to advance the tier"
        )

    def test_failures_keep_us_at_starting_tier(self):
        c = self._curriculum()
        start = c.get_tier_name()
        for _ in range(10):
            c.record(task_id="task_1", normalized_reward=-0.4, resolved=False)
        assert c.get_tier_name() == start

    def test_pick_task_id_returns_pool_member(self):
        c = self._curriculum()
        for _ in range(20):
            tid = c.pick_task_id("easy")
            assert tid in {"task_1", "task_2"}

    def test_weak_spots_drive_pick_when_present(self):
        c = self._curriculum()
        # task_2 looks like a weak spot — multiple low scores; task_1 looks
        # mastered. Give the curriculum enough samples so the weak-spot
        # detector triggers (needs len(scores) >= 2).
        for _ in range(4):
            c.record(task_id="task_1", normalized_reward=0.9, resolved=True)
            c.record(task_id="task_2", normalized_reward=-0.2, resolved=False)
        assert "task_2" in c.get_weak_spots()
        assert "task_1" not in c.get_weak_spots()

    def test_pick_task_id_returns_empty_for_unknown_stage(self):
        c = self._curriculum()
        assert c.pick_task_id("nonexistent") == ""


# =============================================================================
# 2) RewardLogger — single canonical CSV + JSONL
# =============================================================================

class TestRewardLogger:
    def test_creates_csv_with_header_on_init(self, tmp_path):
        RewardLogger(tmp_path)
        csv_path = tmp_path / "reward_log.csv"
        assert csv_path.exists()
        with open(csv_path) as f:
            header = next(csv.reader(f))
        assert "episode" in header
        assert "total_reward" in header
        assert "stack_healthy" in header

    def test_log_appends_one_csv_row_and_one_jsonl_line(self, tmp_path):
        rl = RewardLogger(tmp_path)
        rl.log({
            "stage": "easy", "task_id": "task_1", "tier": "easy",
            "total_reward": 1.5, "diagnosis_reward": 0.4, "fix_reward": 0.6,
            "terminal_reward": 0.5, "num_steps": 4, "is_resolved": True,
            "stack_healthy": True, "last_sla_status": "OK", "wallclock_s": 12.3,
        })
        rl.log({
            "stage": "easy", "task_id": "task_2", "tier": "easy",
            "total_reward": -2.0, "diagnosis_reward": 0.0, "fix_reward": 0.0,
            "terminal_reward": -2.0, "num_steps": 8, "is_resolved": False,
            "stack_healthy": False, "last_sla_status": "VIOLATED",
            "wallclock_s": 99.0,
        })

        with open(rl.csv_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert rows[0]["episode"] == "1"
        assert rows[1]["episode"] == "2"
        assert rows[0]["task_id"] == "task_1"
        assert rows[1]["task_id"] == "task_2"
        assert rows[0]["stack_healthy"] == "1"
        assert rows[1]["stack_healthy"] == "0"

        with open(rl.jsonl_path) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert len(lines) == 2
        assert lines[0]["task_id"] == "task_1"
        assert lines[1]["task_id"] == "task_2"
        # The logger must inject episode + timestamp.
        assert lines[0]["episode"] == 1
        assert "timestamp" in lines[1]

    def test_unknown_stack_healthy_serialized_as_blank(self, tmp_path):
        rl = RewardLogger(tmp_path)
        rl.log({
            "stage": "easy", "task_id": "t", "total_reward": 0.0,
            "diagnosis_reward": 0.0, "fix_reward": 0.0, "terminal_reward": 0.0,
            "num_steps": 0, "is_resolved": False,
            "stack_healthy": None,  # unknown
            "last_sla_status": "OK", "wallclock_s": 0.0,
        })
        with open(rl.csv_path) as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["stack_healthy"] == ""


# =============================================================================
# 3) make_reward_total — single primary reward, no_tool_penalty branch
# =============================================================================

class TestMakeRewardTotal:
    def test_no_tool_penalty_when_trajectory_empty(self):
        reward_total = make_reward_total(no_tool_penalty=-0.5)
        envs = [_FakeEnv(trajectory=[], total_reward=0.0)]
        rewards = reward_total(completions=[None], environments=envs)
        assert rewards == [-0.5]
        # Even on the empty-trajectory path, we still want the logger /
        # curriculum to see the failed episode (so the success rate isn't
        # silently inflated by skipped logs).
        assert envs[0].log_calls == 1

    def test_uses_total_reward_when_trajectory_nonempty(self):
        reward_total = make_reward_total(no_tool_penalty=-0.5)
        env = _FakeEnv(
            trajectory=[{"reward": 0.2}, {"reward": -0.1}],
            total_reward=2.7,
        )
        rewards = reward_total(completions=[None], environments=[env])
        assert rewards == [2.7]
        assert env.log_calls == 1

    def test_no_environments_returns_zeros_and_does_not_crash(self):
        reward_total = make_reward_total()
        assert reward_total(completions=[None, None], environments=[]) == [0.0, 0.0]

    def test_diagnosis_and_fix_metrics_extract_buckets(self):
        env = _FakeEnv(
            trajectory=[{"reward": 0.1}],
            total_reward=2.0,
            diagnosis_reward=1.4,
            fix_reward=0.6,
        )
        diag = reward_diagnosis_metric(completions=[None], environments=[env])
        fix = reward_fix_metric(completions=[None], environments=[env])
        assert diag == [1.4]
        assert fix == [0.6]
        # These are *metrics*, not primary rewards — they must NOT trigger
        # the per-episode logger (only reward_total does).
        assert env.log_calls == 0

    def test_logger_called_exactly_once_per_episode(self):
        reward_total = make_reward_total(no_tool_penalty=-0.5)
        env = _FakeEnv(trajectory=[{"reward": 0.0}], total_reward=0.0)
        reward_total(completions=[None], environments=[env])
        reward_total(completions=[None], environments=[env])
        # Two calls within the same episode — but the env's idempotency flag
        # lives in maybe_log_and_record() in the real wrapper. The fake
        # delegates to its own counter, so verify reward_total *forwards*
        # exactly one call per invocation (idempotency is asserted in the
        # PageZeroToolEnv test below).
        assert env.log_calls == 2


# =============================================================================
# 3b) build_grpo_dataset — per-row EPISODE_BRIEF + episode_task_id column
# =============================================================================


class TestBuildGrpoDataset:
    def test_groups_rows_by_task_for_grpo_group(self):
        ds = build_grpo_dataset(10, ["task_1", "task_2"], group_size=4)
        assert ds.column_names == ["prompt", "episode_task_id"]
        assert [ds[i]["episode_task_id"] for i in range(10)] == [
            "task_1", "task_1", "task_1", "task_1",
            "task_2", "task_2", "task_2", "task_2",
            "task_1", "task_1",
        ]
        u0 = ds[0]["prompt"][-1]["content"]
        assert "EPISODE_BRIEF" in u0
        assert "task_1" in u0
        assert TASK_ALERT_ONE_LINER["task_1"] in u0

    def test_group_size_one_behaves_like_rowwise_cycle(self):
        ds = build_grpo_dataset(5, ["task_1", "task_2"], group_size=1)
        assert [ds[i]["episode_task_id"] for i in range(5)] == [
            "task_1", "task_2", "task_1", "task_2", "task_1"
        ]

    def test_empty_task_pool_raises(self):
        with pytest.raises(ValueError):
            build_grpo_dataset(3, [])

    def test_non_positive_group_size_raises(self):
        with pytest.raises(ValueError):
            build_grpo_dataset(3, ["task_1"], group_size=0)


# =============================================================================
# 4) PageZeroToolEnv._split_terminal + trajectory_payload
# =============================================================================

class TestPageZeroToolEnvPure:
    def _env(self):
        # Build a wrapper without going through __init__ (which constructs a
        # PageZeroEnvClient and connects to a real WebSocket). We only need
        # the pure-Python helpers.
        env = PageZeroToolEnv.__new__(PageZeroToolEnv)
        env.trajectory = []
        env.total_reward = 0.0
        env.diagnosis_reward = 0.0
        env.fix_reward = 0.0
        env.terminal_reward = 0.0
        env.is_done = False
        env._episode_logged = False
        env.task_id = "task_1"
        env.scenario_name = "scenario"
        env.last_sla_status = "OK"
        env.is_resolved = False
        env.stack_healthy = None
        env.start_ts = 0.0
        env.end_ts = 1.5
        env._last_command = ""
        env._last_feedback = ""
        env._last_output_snippet = ""
        env._last_reward = 0.0
        return env

    def test_split_terminal_passthrough_when_not_done(self):
        env = self._env()
        per_step, terminal = env._split_terminal(0.4, was_done=False, canonical_final=None)
        assert per_step == 0.4
        assert terminal == 0.0

    def test_split_terminal_uses_canonical_final_when_done(self):
        env = self._env()
        per_step, terminal = env._split_terminal(
            reward=1.5, was_done=True, canonical_final=0.9
        )
        # terminal estimate = clamp(2 * 0.9 - 1, -1, 1) = 0.8
        assert terminal == pytest.approx(0.8)
        assert per_step == pytest.approx(1.5 - 0.8)

    def test_split_terminal_clamps_to_unit_range(self):
        env = self._env()
        _, terminal = env._split_terminal(reward=10.0, was_done=True, canonical_final=999.0)
        assert terminal == 1.0
        _, terminal = env._split_terminal(reward=-10.0, was_done=True, canonical_final=-999.0)
        assert terminal == -1.0

    def test_trajectory_payload_normalized_reward_is_sum_over_steps(self):
        env = self._env()
        env.trajectory = [
            {"reward": 0.2, "tool": "check_alerts"},
            {"reward": 0.4, "tool": "pg_stat_activity"},
            {"reward": -0.1, "tool": "redis_info"},
        ]
        env.total_reward = 0.5
        env.is_resolved = True
        env.stack_healthy = True
        payload = env.trajectory_payload()
        assert payload["num_steps"] == 3
        assert payload["normalized_reward"] == pytest.approx(0.5 / 3)
        assert payload["total_reward"] == pytest.approx(0.5)
        assert payload["stack_healthy"] is True
        assert payload["tool_sequence"] == ["check_alerts", "pg_stat_activity", "redis_info"]

    def test_trajectory_payload_handles_zero_step_episode(self):
        env = self._env()
        payload = env.trajectory_payload()
        assert payload["num_steps"] == 0
        assert payload["normalized_reward"] == 0.0
        assert payload["tool_sequence"] == []

    def test_maybe_log_and_record_is_idempotent(self, tmp_path):
        env = self._env()
        env.trajectory = [{"reward": 0.1, "tool": "check_alerts"}]
        env.total_reward = 0.1

        rl = RewardLogger(tmp_path)
        PageZeroToolEnv.REWARD_LOGGER = rl
        try:
            env.maybe_log_and_record()
            env.maybe_log_and_record()
            env.maybe_log_and_record()
        finally:
            PageZeroToolEnv.REWARD_LOGGER = None

        with open(rl.csv_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1, "maybe_log_and_record must log only once per episode"

    def test_format_observation_includes_incident_context(self):
        from types import SimpleNamespace

        env = self._env()
        env.task_id = "task_3"
        env.scenario_name = "Redis OOM headline"
        obs = SimpleNamespace(
            tool_output="🚨 ALERT: Redis memory",
            active_alerts=["Redis memory"],
            sla_status="OK",
            revenue_loss_usd=0.0,
            downtime_minutes=0.0,
            step=0,
            max_steps=15,
            hint=None,
            phase=None,
            is_fix_step=False,
            repeat_count=1,
            judge_feedback=None,
        )
        text = env._format_observation(obs, reward=0.0)
        assert "INCIDENT CONTEXT:" in text
        assert "task_3" in text
        assert "Redis OOM headline" in text


# =============================================================================
# 5) Persona selection
# =============================================================================

class TestPersonaForDifficulty:
    @pytest.mark.parametrize(
        "difficulty,expected",
        [
            (0.0, "junior"),
            (0.39, "junior"),
            (0.4, "senior"),
            (0.69, "senior"),
            (0.7, "principal"),
            (1.0, "principal"),
        ],
    )
    def test_persona_thresholds(self, difficulty, expected):
        assert persona_for_difficulty(difficulty) == expected


# =============================================================================
# 6) Deterministic phase shaping
# =============================================================================

class TestPhaseShaping:
    def test_first_action_triage_gets_bonus(self):
        # phase_score expects ``history`` to include the *current* step at
        # the very end (env convention).
        history = [{"tool": "check_alerts", "output": "..."}]
        assert phase_score("triage", history) == REWARD_TRIAGE_FIRST

    def test_first_action_non_triage_gets_penalty(self):
        history = [{"tool": "pg_cancel_query", "output": "..."}]
        assert phase_score("fix", history) == REWARD_WRONG_FIRST

    def test_correct_order_advances_one_phase(self):
        # triage → investigate (one-phase advance)
        history = [
            {"tool": "check_alerts", "output": "..."},
            {"tool": "pg_stat_activity", "output": "..."},
        ]
        assert phase_score("investigate", history) == REWARD_CORRECT_ORDER

    def test_skipping_phases_is_penalized(self):
        # triage → fix (skips investigate + diagnose)
        history = [
            {"tool": "check_alerts", "output": "..."},
            {"tool": "pg_cancel_query", "output": "..."},
        ]
        assert phase_score("fix", history) == REWARD_SKIPPED_PHASE

    def test_going_backward_is_penalized(self):
        # fix → investigate (without an intervening verify)
        history = [
            {"tool": "check_alerts", "output": "..."},
            {"tool": "pg_stat_activity", "output": "..."},
            {"tool": "pg_explain_analyze", "output": "..."},
            {"tool": "pg_cancel_query", "output": "..."},
            {"tool": "pg_stat_activity", "output": "..."},
        ]
        assert phase_score("investigate", history) == REWARD_BACKWARD_PHASE

    def test_get_skipped_phases_lists_intermediate_phases(self):
        history = [
            {"tool": "check_alerts", "output": "..."},
            {"tool": "pg_cancel_query", "output": "..."},
        ]
        skipped = get_skipped_phases("fix", history)
        # Skipped between triage and fix: investigate, diagnose
        assert "investigate" in skipped
        assert "diagnose" in skipped


# =============================================================================
# 7) Integration sanity — make_reward_total + PageZeroToolEnv pipeline
# =============================================================================

class TestEndToEndRewardPipeline:
    def test_no_tool_penalty_persists_through_logger(self, tmp_path):
        """Empty trajectory → -0.5 reward AND a row in the CSV with num_steps=0."""
        env = PageZeroToolEnv.__new__(PageZeroToolEnv)
        env.trajectory = []
        env.total_reward = 0.0
        env.diagnosis_reward = 0.0
        env.fix_reward = 0.0
        env.terminal_reward = 0.0
        env.is_done = False
        env._episode_logged = False
        env.task_id = "task_1"
        env.scenario_name = ""
        env.last_sla_status = "OK"
        env.is_resolved = False
        env.stack_healthy = None
        env.start_ts = 0.0
        env.end_ts = 1.0
        env._last_command = ""
        env._last_feedback = ""
        env._last_output_snippet = ""
        env._last_reward = 0.0

        rl = RewardLogger(tmp_path)
        PageZeroToolEnv.REWARD_LOGGER = rl
        PageZeroToolEnv.CURRICULUM = MasteryCurriculum(
            stage_task_pools={"easy": ["task_1"]}
        )
        try:
            reward_total = make_reward_total(no_tool_penalty=-0.5)
            rewards = reward_total(completions=[None], environments=[env])
        finally:
            PageZeroToolEnv.REWARD_LOGGER = None
            PageZeroToolEnv.CURRICULUM = None

        assert rewards == [-0.5]
        with open(rl.csv_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["num_steps"] == "0"
        assert rows[0]["is_resolved"] == "0"
        # Total reward in CSV reflects the env.total_reward (0.0) — the
        # -0.5 is the *training* signal, not the ledger of what the agent
        # actually earned in the env. We log both so audits stay honest.
        assert float(rows[0]["total_reward"]) == 0.0
