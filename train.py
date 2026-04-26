"""
GRPO Training Script — PageZero SRE Agent

Canonical, kube-sre-gym-aligned wrapper. Notebook and CLI now share the
same wrapper logic (accumulating reward buckets, prior-feedback injection,
top-level reward CSV, mastery-gated curriculum).

Setup (2 terminals on H100):

  # Install
  pip install -e ".[train]"

  # Terminal 1: OpenEnv server (adversarial mode with Claude judge)
  GYM_MODE=adversarial LLM_BACKEND=anthropic ANTHROPIC_API_KEY=sk-ant-... uv run server

  # Terminal 2: GRPO training
  python train.py --vllm-mode colocate
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import threading
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Help PyTorch reuse fragmented GPU memory (critical for TRL+vLLM colocate on 80GB)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from datasets import Dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

try:
    from .client import PageZeroEnvClient
    from .models import PageZeroAction
except (ImportError, ValueError):
    from client import PageZeroEnvClient
    from models import PageZeroAction


# Tools we treat as fix attempts; mirror server/llm_judge.FIX_TOOLS so the
# wrapper never drifts from the env's auto-detect set.
_FIX_TOOL_NAMES = frozenset({
    "pg_cancel_query", "pg_create_index", "pg_vacuum",
    "redis_flush_db", "docker_restart", "rollback_deploy",
})


# ---- Optional TRL/vLLM compatibility patch for older stacks ----
_orig_vllm_gen = None


def _patch_vllm_generate(trainer: GRPOTrainer) -> None:
    """Wrap vLLM generate to normalize logprobs shape on older TRL stacks."""
    global _orig_vllm_gen
    if _orig_vllm_gen is not None or not hasattr(trainer, "vllm_generation"):
        return

    _orig_vllm_gen = trainer.vllm_generation.generate

    def _wrapped_generate(**kwargs):
        result = _orig_vllm_gen(**kwargs)
        prompt_ids, completion_ids, logprobs, *rest = result
        if logprobs and logprobs[0] and isinstance(logprobs[0][0], float):
            logprobs = [[[lp] for lp in seq] for seq in logprobs]
        return (prompt_ids, completion_ids, logprobs, *rest)

    trainer.vllm_generation.generate = _wrapped_generate


def patch_trl_vllm_compat() -> None:
    """Apply TRL/vLLM compatibility patches if needed by this TRL build."""
    _orig_train = GRPOTrainer.train

    def _patched_train(self, *args, **kwargs):
        _patch_vllm_generate(self)
        return _orig_train(self, *args, **kwargs)

    GRPOTrainer.train = _patched_train


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a Staff SRE on-call. Diagnose and fix the cascading incident across the Application, PostgreSQL, and Redis cache.

To use a tool, you MUST use this exact format:
<tool_call>
{"name": "check_alerts", "arguments": {}}
</tool_call>

## Tool cheat-sheet (when to use what)
- **Triage (first step almost always here):** `check_alerts` — container/app health snapshot. Do not skip for generic “what is on fire?” context.
- **Postgres CPU / latency / connections:** prefer `pg_stat_activity`, `pg_locks`, `pg_stat_statements`, `pg_explain_analyze` before any fix. Fixes: `pg_cancel_query`, `pg_create_index`, `pg_vacuum`.
- **Redis memory / OOM / cache:** prefer `redis_info`, `redis_keys`, `redis_slowlog` before `redis_flush_db`.
- **App / HTTP errors:** `docker_logs`, `docker_ps`, `read_app_logs`, `curl_endpoint`.
- **Disk full scenarios (only when alert mentions disk / WAL / space):** `check_disk_usage`. Do NOT use disk checks for pure Redis-memory or Postgres-query alerts.
- **Documentation (late phase):** `diagnose_root_cause` and `write_postmortem` only after you have evidence from investigation tools (and usually after attempted fixes). Never use them as a substitute for running commands.
- **Termination:** `done` only when the stack is verified healthy.

## Gold pattern (PostgreSQL runaway query — imitate the structure, not hard-coded IDs)
Turn 1 — triage:
<tool_call>
{"name": "check_alerts", "arguments": {}}
</tool_call>
Turn 2 — investigate DB:
<tool_call>
{"name": "pg_stat_activity", "arguments": {}}
</tool_call>
Turn 3 — fix (use a real pid from step 2 output when you run for real):
<tool_call>
{"name": "pg_cancel_query", "arguments": {"pid": 12345}}
</tool_call>
Turn 4 — verify:
<tool_call>
{"name": "pg_stat_activity", "arguments": {}}
</tool_call>
Then continue until healthy; call `done` last.

Each tool result includes a STEP REWARD and (when available) JUDGE FEEDBACK — read both before choosing your next action.
"""

# One-line alert text keyed by OpenEnv task_id (matches ``server/llm_designer.py``
# scenario order: warmup×5, medium×5, hard×2). Shown in the dataset *user*
# row so tiny models see the incident before the first generation token.
TASK_ALERT_ONE_LINER: Dict[str, str] = {
    "task_1": (
        "CRITICAL: API p99 latency > 5s, PostgreSQL CPU at 95% — "
        "investigate active queries in PostgreSQL"
    ),
    "task_2": (
        "WARNING: /api/orders endpoint p99 > 3s — sequential scans suspected on orders table"
    ),
    "task_3": (
        "CRITICAL: Redis memory usage > 95%, OOM errors in app logs — "
        "check redis_info and flush orphaned keys"
    ),
    "task_4": (
        "WARNING: Cache miss rate spikes to 100% — Redis may have been flushed; check redis_info"
    ),
    "task_5": (
        "CRITICAL: 'too many connections' errors, new requests timing out — "
        "check pg_stat_activity for idle connections"
    ),
    "task_6": (
        "CRITICAL: PostgreSQL write errors — 'could not write to file'; "
        "disk usage may be at 100% — run check_disk_usage"
    ),
    "task_7": (
        "CRITICAL: pagezero-app-1 returning HTTP 502 — container may have crashed; run docker_ps"
    ),
    "task_8": (
        "WARNING: /api/orders query degraded 4x — dead tuple bloat suspected; try pg_vacuum"
    ),
    "task_9": (
        "CRITICAL: App HTTP 500 — DB logs show 'permission denied for table orders'"
    ),
    "task_10": (
        "CRITICAL: API error rate 40%, cache hit 0%, DB CPU 98% — "
        "check redis_info then pg_stat_activity"
    ),
    "task_11": "CRITICAL: API Latency > 10s, entire application unresponsive.",
    "task_12": "CRITICAL: Multiple services degraded — app 503s, DB locks, Redis OOM.",
}


def build_grpo_dataset(num_rows: int, stage_task_ids: List[str], group_size: int = 4) -> Dataset:
    """HF dataset rows for GRPO + ``environment_factory``.

    Each row includes:
      * ``prompt`` — system + user, with **EPISODE_BRIEF** (task_id + alert line).
      * ``episode_task_id`` — passed by TRL into ``PageZeroToolEnv.reset(**kwargs)``
        so the env reset matches the user row (small models see the same id twice).

    Args:
        num_rows: Upper-bound episode count / dataset length for this stage.
        stage_task_ids: Curriculum task pool for the current stage (non-empty).
        group_size: Number of candidates in one GRPO comparison group. Rows are
            emitted in contiguous same-task blocks of this size (typically
            ``NUM_GENERATIONS``).

    Returns:
        A :class:`datasets.Dataset` with columns ``prompt`` and ``episode_task_id``.
    """
    if not stage_task_ids:
        raise ValueError("build_grpo_dataset: stage_task_ids must be non-empty")
    if int(group_size) <= 0:
        raise ValueError("build_grpo_dataset: group_size must be > 0")

    group_size = int(group_size)
    prompts: List[List[Dict[str, str]]] = []
    episode_task_ids: List[str] = []
    for i in range(int(num_rows)):
        # Keep one task id per GRPO group so relative advantages compare
        # alternative tool plans on the *same* incident.
        gid = i // group_size
        tid = stage_task_ids[gid % len(stage_task_ids)]
        headline = TASK_ALERT_ONE_LINER.get(
            tid,
            "Read TOOL OUTPUT after reset for the live alert text.",
        )
        user_msg = (
            "Diagnose and fix this production incident.\n\n"
            "EPISODE_BRIEF (your tools must match this assignment):\n"
            f"  task_id: {tid}\n"
            f"  alert_summary: {headline}\n"
        )
        prompts.append(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]
        )
        episode_task_ids.append(tid)

    return Dataset.from_dict({"prompt": prompts, "episode_task_id": episode_task_ids})


# =============================================================================
# Mastery-gated curriculum (kube-sre-gym style)
# =============================================================================

DIFFICULTY_TIERS: List[Dict[str, Any]] = [
    {"name": "warmup",       "max_diff": 0.25, "min_episodes": 5,  "advance_rate": 0.6},
    {"name": "beginner",     "max_diff": 0.40, "min_episodes": 5,  "advance_rate": 0.6},
    {"name": "intermediate", "max_diff": 0.60, "min_episodes": 8,  "advance_rate": 0.65},
    {"name": "advanced",     "max_diff": 0.80, "min_episodes": 10, "advance_rate": 0.7},
    {"name": "expert",       "max_diff": 0.95, "min_episodes": 0,  "advance_rate": 1.0},
]


class MasteryCurriculum:
    """Tier-gated curriculum with fast-track and per-task mastery tracking.

    The optimizer cares about ``recent_success_rate`` (over a sliding window
    of episodes) being above ``advance_rate`` — *not* a fixed episode count.
    Per-task scores bias sampling toward weak spots so easy tasks the agent
    has already solved stop dominating the rollout budget.
    """

    SUCCESS_THRESHOLD = 0.0   # normalized reward > 0 counts as success
    MASTERY_THRESHOLD = 0.6
    MASTERY_WINDOW = 8

    def __init__(
        self,
        stage_task_pools: Dict[str, List[str]] | None = None,
        success_threshold: float | None = None,
    ):
        self.stage_task_pools = stage_task_pools or {}
        self.history: deque[bool] = deque(maxlen=20)
        self.recent_episodes_in_tier = 0
        self._tier_index = 0
        self.task_scores: Dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=self.MASTERY_WINDOW)
        )
        if success_threshold is not None:
            self.SUCCESS_THRESHOLD = success_threshold

    @property
    def tier(self) -> Dict[str, Any]:
        return DIFFICULTY_TIERS[self._tier_index]

    def get_tier_name(self) -> str:
        return self.tier["name"]

    def get_difficulty(self) -> float:
        return self.tier["max_diff"]

    def record(
        self,
        task_id: str,
        normalized_reward: float,
        resolved: bool,
        fix_reward: float = 0.0,
        num_steps: int = 0,
    ) -> None:
        # Count a non-terminal episode as curriculum-success only when it
        # contains a real fix attempt and enough interaction depth.
        success = bool(
            resolved
            or (
                normalized_reward > self.SUCCESS_THRESHOLD
                and float(fix_reward) > 0.0
                and int(num_steps) >= 3
            )
        )
        self.history.append(success)
        self.recent_episodes_in_tier += 1
        if task_id:
            self.task_scores[task_id].append(float(normalized_reward))
        self._maybe_advance()

    def _maybe_advance(self) -> None:
        if self._tier_index >= len(DIFFICULTY_TIERS) - 1:
            return
        tier = self.tier
        rate = self._recent_success_rate()
        # kube-sre-gym fast-track: 90%+ over 3 episodes → advance immediately.
        fast_track = (self.recent_episodes_in_tier >= 3 and rate >= 0.9)
        if not fast_track and self.recent_episodes_in_tier < tier["min_episodes"]:
            return
        if rate >= tier["advance_rate"]:
            logger.info(
                f"[curriculum] advancing from {tier['name']} → "
                f"{DIFFICULTY_TIERS[self._tier_index + 1]['name']} "
                f"(rate={rate:.0%}, episodes={self.recent_episodes_in_tier}"
                f"{', FAST-TRACK' if fast_track else ''})"
            )
            self._tier_index += 1
            self.recent_episodes_in_tier = 0

    def _recent_success_rate(self) -> float:
        if not self.history:
            return 0.0
        return sum(self.history) / len(self.history)

    def get_weak_spots(self) -> List[str]:
        """Tasks whose mean normalized reward is below mastery."""
        weak: List[str] = []
        for tid, scores in self.task_scores.items():
            if len(scores) >= 2 and (sum(scores) / len(scores)) < self.MASTERY_THRESHOLD:
                weak.append(tid)
        return weak

    def pick_task_id(self, stage_name: str) -> str:
        """Sample a task id from the stage pool, biasing toward weak spots."""
        import random
        pool = self.stage_task_pools.get(stage_name, [])
        if not pool:
            return ""
        weak = [tid for tid in self.get_weak_spots() if tid in pool]
        # 70% weak-spot, 30% uniform — stay exploratory even when weak set is non-empty.
        if weak and random.random() < 0.7:
            return random.choice(weak)
        return random.choice(pool)


# =============================================================================
# Top-level reward / trajectory logger (single CSV at output dir root)
# =============================================================================

REWARD_LOG_HEADER = [
    "episode", "stage", "task_id", "tier", "total_reward",
    "diagnosis_reward", "fix_reward", "terminal_reward",
    "num_steps", "is_resolved", "stack_healthy",
    "last_sla_status", "wallclock_s", "timestamp",
]


class RewardLogger:
    """Thread-safe per-episode logger that writes a top-level CSV + JSONL."""

    def __init__(self, output_dir: Path, csv_name: str = "reward_log.csv",
                 jsonl_name: str = "trajectories.jsonl") -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir
        self.csv_path = output_dir / csv_name
        self.jsonl_path = output_dir / jsonl_name
        self.episode_counter = 0
        self.totals: List[float] = []
        self.resolved_flags: List[bool] = []
        self._lock = threading.Lock()
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow(REWARD_LOG_HEADER)
        # Truncate any previous JSONL from a stale run in this dir.
        self.jsonl_path.touch(exist_ok=True)

    def log(self, payload: Dict[str, Any]) -> None:
        with self._lock:
            self.episode_counter += 1
            payload = dict(payload)
            payload["episode"] = self.episode_counter
            payload["timestamp"] = datetime.now().isoformat()
            self.totals.append(float(payload.get("total_reward", 0.0)))
            self.resolved_flags.append(bool(payload.get("is_resolved", False)))

            with open(self.csv_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    payload["episode"],
                    payload.get("stage", ""),
                    payload.get("task_id", ""),
                    payload.get("tier", ""),
                    payload.get("total_reward", 0.0),
                    payload.get("diagnosis_reward", 0.0),
                    payload.get("fix_reward", 0.0),
                    payload.get("terminal_reward", 0.0),
                    payload.get("num_steps", 0),
                    int(bool(payload.get("is_resolved", False))),
                    "" if payload.get("stack_healthy") is None else int(bool(payload["stack_healthy"])),
                    payload.get("last_sla_status", ""),
                    round(float(payload.get("wallclock_s", 0.0)), 2),
                    payload["timestamp"],
                ])

            with open(self.jsonl_path, "a") as f:
                f.write(json.dumps(payload, default=str) + "\n")

            n = len(self.totals)
            mean_all = sum(self.totals) / n
            window = self.totals[-10:]
            mean_10 = sum(window) / len(window)
            resolved_rate = sum(self.resolved_flags[-10:]) / len(self.resolved_flags[-10:])
            logger.info(
                f"Episode {payload['episode']} ({payload.get('stage','?')}/"
                f"{payload.get('task_id','?')}): "
                f"reward={payload['total_reward']:+.2f} "
                f"(diag={payload.get('diagnosis_reward',0):+.2f}, "
                f"fix={payload.get('fix_reward',0):+.2f}, "
                f"term={payload.get('terminal_reward',0):+.2f}) | "
                f"steps={payload.get('num_steps')} resolved={payload.get('is_resolved')} | "
                f"mean(10)={mean_10:+.2f} resolved_rate(10)={resolved_rate:.0%}"
            )


# =============================================================================
# Canonical PageZeroToolEnv wrapper (notebook + CLI share this)
# =============================================================================

class PageZeroToolEnv:
    """Canonical wrapper for ``GRPOTrainer(environment_factory=...)``.

    Public methods (except ``reset``) are exposed as tools automatically.

    Per-step it:
      * splits server reward into ``per_step`` + ``terminal`` components
        (so credit assignment is preserved for logging),
      * accumulates into ``diagnosis_reward`` / ``fix_reward`` / ``terminal_reward``
        (the previous train.py wrapper *overwrote* these — bug),
      * captures a structured trajectory used by the top-level logger,
      * injects prior turn's command + output + reward + judge feedback
        into the *next* tool result, giving the agent in-episode teacher
        signal (so it doesn't have to wait for the gradient to discover
        what worked).
    """

    BASE_URL: str = "http://localhost:8000"
    REWARD_LOGGER: Optional[RewardLogger] = None
    CURRICULUM: Optional[MasteryCurriculum] = None
    STAGE: str = "default"
    STAGE_TASK_IDS: List[str] = []
    STRICT_EPISODE_TASK_BINDING: bool = True
    NO_TOOL_PENALTY: float = -0.5

    _instances: List["PageZeroToolEnv"] = []

    def __init__(self) -> None:
        """Create a new tool env wrapper and register it for ``_close_all()``."""
        self.client = PageZeroEnvClient(base_url=self.BASE_URL)
        self.total_reward = 0.0
        self.diagnosis_reward = 0.0
        self.fix_reward = 0.0
        self.terminal_reward = 0.0
        self.is_done = False
        self._episode_logged = False
        self.trajectory: List[Dict[str, Any]] = []
        self.task_id: str = ""
        self.scenario_name: str = ""
        self.last_sla_status: str = "OK"
        self.is_resolved: bool = False
        self.stack_healthy: Optional[bool] = None
        self.start_ts: float = 0.0
        self.end_ts: Optional[float] = None
        self._last_feedback: str = ""
        self._last_command: str = ""
        self._last_output_snippet: str = ""
        self._last_reward: float = 0.0
        self.__class__._instances.append(self)

    # ---- Curriculum hooks (set by the training driver before each stage) ----
    # NOTE: names are underscore-prefixed so TRL/HF ``get_json_schema`` does not
    # treat them as agent-callable tools (classmethods like ``set_stage`` were
    # incorrectly scanned and required brittle Google-style docstrings).
    @classmethod
    def _set_stage(cls, name: str, task_ids: List[str]) -> None:
        """Bind the curriculum stage label and the task-id pool used by ``reset()``."""
        cls.STAGE = name
        cls.STAGE_TASK_IDS = list(task_ids)

    @classmethod
    def _close_all(cls) -> None:
        """Close all live environment clients created by this wrapper class."""
        for env in cls._instances:
            try:
                env.client.close()
            except Exception:
                pass
        cls._instances.clear()

    # ---- Lifecycle ----
    def reset(self, **kwargs: Any) -> str | None:
        """Start a new episode and return the initial observation text for the model.

        Args:
            kwargs: TRL passes the dataset row here. We consume ``episode_task_id``
                (must match ``STAGE_TASK_IDS``) and ``prompt``; other keys are ignored.
                When ``episode_task_id`` is present and valid, it overrides curriculum
                sampling so the env matches the ``EPISODE_BRIEF`` in the user message.

        Returns:
            Formatted observation string appended to the user prompt, or ``None``
            if the integration passes no observation (rare).
        """
        # TRL passes the full dataset row into ``reset``; align reset(task_id)
        # with ``episode_task_id`` from ``build_grpo_dataset`` when it matches
        # the active stage pool (so the user EPISODE_BRIEF and env stay consistent).
        kwargs.pop("prompt", None)
        row_tid = kwargs.pop("episode_task_id", None)
        row_tid_s = str(row_tid).strip() if row_tid is not None else ""

        pool = [str(x) for x in (self.STAGE_TASK_IDS or [])]
        chosen_task_id = ""
        if row_tid_s and pool and row_tid_s in pool:
            chosen_task_id = row_tid_s
        elif row_tid_s and pool and self.STRICT_EPISODE_TASK_BINDING:
            raise ValueError(
                f"episode_task_id={row_tid_s!r} is not in active stage pool={pool}"
            )
        elif not row_tid_s and pool and self.STRICT_EPISODE_TASK_BINDING:
            raise ValueError(
                "episode_task_id is required for deterministic grouping but missing from dataset row"
            )
        elif self.CURRICULUM is not None:
            chosen_task_id = self.CURRICULUM.pick_task_id(self.STAGE)
        elif pool:
            import random
            chosen_task_id = random.choice(pool)

        logger.info(
            "[reset] stage=%s row_task_id=%s chosen_task_id=%s strict=%s",
            self.STAGE,
            (row_tid_s or "<missing>"),
            (chosen_task_id or "<auto>"),
            self.STRICT_EPISODE_TASK_BINDING,
        )

        try:
            result = self.client.reset(task_id=chosen_task_id) if chosen_task_id else self.client.reset()
        except TypeError:
            result = self.client.reset()
            chosen_task_id = ""

        self.total_reward = 0.0
        self.diagnosis_reward = 0.0
        self.fix_reward = 0.0
        self.terminal_reward = 0.0
        self.is_done = bool(result.done)
        self._episode_logged = False
        self.trajectory = []
        self.task_id = chosen_task_id
        self.scenario_name = (getattr(result.observation, "tool_output", "") or "")[:120]
        self.last_sla_status = getattr(result.observation, "sla_status", "OK") or "OK"
        self.is_resolved = False
        self.stack_healthy = None
        self.start_ts = datetime.now().timestamp()
        self.end_ts = None
        self._last_feedback = ""
        self._last_command = ""
        self._last_output_snippet = ""
        self._last_reward = 0.0
        return self._format_observation(result.observation, reward=0.0)

    def _format_observation(self, obs, reward: float) -> str:
        tool_output = getattr(obs, "tool_output", "") or ""
        alerts = getattr(obs, "active_alerts", []) or []
        alert_text = "\n".join(alerts) if alerts else "None"
        sla_status = getattr(obs, "sla_status", "OK")
        revenue_loss = getattr(obs, "revenue_loss_usd", 0.0)
        downtime_minutes = getattr(obs, "downtime_minutes", 0.0)
        step = getattr(obs, "step", 0)
        max_steps = getattr(obs, "max_steps", 15)
        hint = getattr(obs, "hint", "") or getattr(obs, "judge_feedback", "") or ""
        phase = getattr(obs, "phase", None)
        is_fix_step = getattr(obs, "is_fix_step", False)
        repeat_count = getattr(obs, "repeat_count", 1)

        # In-episode teacher signal: surface the *previous* turn's outcome so
        # the agent can update its plan without waiting for a gradient.
        prior_block = ""
        if self._last_command:
            prior_block = (
                "PREVIOUS ACTION RECAP:\n"
                f"  command: {self._last_command}\n"
                f"  reward : {self._last_reward:+.3f}\n"
                f"  judge  : {self._last_feedback[:240]}\n"
                f"  output : {self._last_output_snippet[:240]}\n\n"
            )

        # Small models under-condition on free-text alerts; the static dataset
        # user line is generic ("Diagnose and fix…"), so surface curriculum
        # identity explicitly every turn (cheap tokens, clearer credit assignment).
        incident_header = ""
        if self.task_id or self.scenario_name:
            incident_header = (
                "INCIDENT CONTEXT:\n"
                f"  task_id: {self.task_id or 'auto'}\n"
                f"  scenario: {self.scenario_name or '(unknown)'}\n\n"
            )

        text = (
            f"{prior_block}"
            f"{incident_header}"
            f"TOOL OUTPUT:\n{tool_output}\n\n"
            f"CURRENT ALERTS:\n{alert_text}\n\n"
            f"SLA STATUS: {sla_status}\n"
            f"REVENUE LOST: ${revenue_loss}\n"
            f"DOWNTIME: {downtime_minutes} minutes\n"
            f"PHASE: {phase} (fix_step={is_fix_step}, repeat={repeat_count})\n"
            f"STEP REWARD: {reward:+.4f}\n"
            f"STEP: {step}/{max_steps}"
        )
        if hint:
            text += f"\nJUDGE FEEDBACK: {hint}"
        return text

    def _split_terminal(self, reward: float, was_done: bool, canonical_final: float | None):
        """Split server reward into per-step + (estimated) terminal component."""
        if not was_done:
            return reward, 0.0
        # If the env returned a canonical 0..1 score, the server-side terminal
        # training contribution is roughly (2 * canonical - 1). Bound it so a
        # noisy estimate cannot dominate the bucket attribution.
        if canonical_final is not None:
            approx = max(-1.0, min(1.0, 2.0 * float(canonical_final) - 1.0))
            return reward - approx, approx
        return reward, 0.0

    def _run_tool(self, tool: str, args: Dict[str, Any]) -> str:
        if self.is_done and tool != "done":
            raise ValueError("Episode already done. No further tools are allowed.")

        result = self.client.step(PageZeroAction(tool=tool, args=args))
        reward = float(result.reward or 0.0)
        was_done = bool(result.done)
        obs = result.observation

        canonical_final = getattr(obs, "final_score", None)
        try:
            canonical_final = float(canonical_final) if canonical_final is not None else None
        except Exception:
            canonical_final = None

        per_step, terminal = self._split_terminal(reward, was_done, canonical_final)
        if was_done:
            self.terminal_reward = terminal

        self.total_reward += reward
        self.is_done = was_done

        is_fix = (tool in _FIX_TOOL_NAMES) or bool(getattr(obs, "is_fix_step", False))
        if is_fix:
            self.fix_reward += per_step
        else:
            self.diagnosis_reward += per_step

        sla_status = getattr(obs, "sla_status", "OK") or "OK"
        self.last_sla_status = sla_status
        snippet = (getattr(obs, "tool_output", "") or "")[:240]

        feedback = (
            getattr(obs, "judge_feedback", None)
            or getattr(obs, "hint", "")
            or ""
        )
        self._last_command = json.dumps({"name": tool, "arguments": args}, default=str)[:200]
        self._last_feedback = str(feedback or "")
        self._last_output_snippet = snippet
        self._last_reward = reward

        self.trajectory.append({
            "step": len(self.trajectory) + 1,
            "tool": tool,
            "args": args,
            "reward": reward,
            "per_step_reward": per_step,
            "terminal_reward_component": terminal if was_done else 0.0,
            "is_fix": is_fix,
            "phase": getattr(obs, "phase", None),
            "repeat_count": getattr(obs, "repeat_count", 1),
            "sla_status": sla_status,
            "is_done": was_done,
            "stack_healthy": getattr(obs, "stack_healthy", None),
            "judge_feedback": str(feedback or "")[:300],
            "output_snippet": snippet,
        })

        if was_done:
            self.end_ts = datetime.now().timestamp()
            self.stack_healthy = getattr(obs, "stack_healthy", None)
            self.is_resolved = bool(self.stack_healthy)

        return self._format_observation(obs, reward=reward)

    def trajectory_payload(self) -> Dict[str, Any]:
        """Build a JSON-serializable summary of the episode for logging / curriculum.

        Returns:
            Dict with rewards, step counts, tool sequence, and full trajectory.
        """
        n_steps = len(self.trajectory)
        # Normalized reward used by the curriculum signal — kube-sre-gym uses
        # raw_sum / steps so a 1-step lucky reward does not look like a
        # successful 5-step trajectory.
        raw_sum = float(sum(t["reward"] for t in self.trajectory))
        normalized = raw_sum / n_steps if n_steps > 0 else 0.0
        return {
            "stage": self.STAGE,
            "task_id": self.task_id,
            "tier": (self.CURRICULUM.get_tier_name() if self.CURRICULUM else ""),
            "scenario_name": self.scenario_name,
            "num_steps": n_steps,
            "is_resolved": bool(self.is_resolved),
            "stack_healthy": self.stack_healthy,
            "total_reward": float(self.total_reward),
            "normalized_reward": normalized,
            "diagnosis_reward": float(self.diagnosis_reward),
            "fix_reward": float(self.fix_reward),
            "terminal_reward": float(self.terminal_reward),
            "last_sla_status": self.last_sla_status,
            "wallclock_s": (self.end_ts or datetime.now().timestamp()) - self.start_ts,
            "tool_sequence": [t["tool"] for t in self.trajectory],
            "trajectory": self.trajectory,
        }

    def maybe_log_and_record(self) -> None:
        """Log trajectory + feed normalized reward into the curriculum once."""
        if self._episode_logged:
            return
        payload = self.trajectory_payload()
        if self.REWARD_LOGGER is not None:
            self.REWARD_LOGGER.log(payload)
        if self.CURRICULUM is not None:
            self.CURRICULUM.record(
                task_id=self.task_id,
                normalized_reward=payload["normalized_reward"],
                resolved=payload["is_resolved"],
                fix_reward=payload.get("fix_reward", 0.0),
                num_steps=payload.get("num_steps", 0),
            )
        self._episode_logged = True

    # =========================================================================
    # Tools (TRL auto-discovers public methods → tool calls)
    # =========================================================================

    # --- Triage ---
    def check_alerts(self) -> str:
        """Check active incident alerts.

        Returns:
            Current alert information.
        """
        return self._run_tool("check_alerts", {})

    def get_service_metrics(self, service: str = "app") -> str:
        """Get service metrics.

        Args:
            service: Service name such as app, redis, or postgres.

        Returns:
            Service metrics.
        """
        return self._run_tool("get_service_metrics", {"service": service})

    def get_error_rate(self) -> str:
        """Get aggregate application error rate.

        Returns:
            Error rate summary.
        """
        return self._run_tool("get_error_rate", {})

    # --- Application ---
    def read_app_logs(self, lines: int = 200) -> str:
        """Read recent application logs.

        Args:
            lines: Number of log lines to fetch.

        Returns:
            Log output.
        """
        return self._run_tool("read_app_logs", {"lines": lines})

    def search_logs(self, pattern: str) -> str:
        """Search logs for a text pattern.

        Args:
            pattern: Search pattern.

        Returns:
            Matching log lines.
        """
        return self._run_tool("search_logs", {"pattern": pattern})

    def get_recent_deploys(self) -> str:
        """List recent deployments.

        Returns:
            Deployment history.
        """
        return self._run_tool("get_recent_deploys", {})

    def rollback_deploy(self) -> str:
        """Rollback latest deployment.

        Returns:
            Rollback status.
        """
        return self._run_tool("rollback_deploy", {})

    def curl_endpoint(self, url: str) -> str:
        """Curl an endpoint for health/behavior check.

        Args:
            url: Endpoint URL.

        Returns:
            HTTP response summary.
        """
        return self._run_tool("curl_endpoint", {"url": url})

    # --- PostgreSQL ---
    def pg_stat_activity(self) -> str:
        """Inspect PostgreSQL active sessions.

        Returns:
            Active query/session information.
        """
        return self._run_tool("pg_stat_activity", {})

    def pg_locks(self) -> str:
        """Inspect PostgreSQL lock state.

        Returns:
            Lock diagnostics.
        """
        return self._run_tool("pg_locks", {})

    def pg_explain_analyze(self, query: str) -> str:
        """Run EXPLAIN ANALYZE on a SQL query.

        Args:
            query: SQL query text.

        Returns:
            Query plan and timing.
        """
        return self._run_tool("pg_explain_analyze", {"query": query})

    def pg_stat_statements(self) -> str:
        """Inspect pg_stat_statements.

        Returns:
            Statement-level performance stats.
        """
        return self._run_tool("pg_stat_statements", {})

    def pg_cancel_query(self, pid: int) -> str:
        """Cancel a PostgreSQL backend query.

        Args:
            pid: Process id to cancel.

        Returns:
            Cancellation result.
        """
        return self._run_tool("pg_cancel_query", {"pid": pid})

    def pg_create_index(self, table: str, column: str) -> str:
        """Create an index on table(column).

        Args:
            table: Table name.
            column: Column name.

        Returns:
            Index creation result.
        """
        return self._run_tool("pg_create_index", {"table": table, "column": column})

    def pg_vacuum(self, table: str) -> str:
        """Run VACUUM on a table.

        Args:
            table: Table name.

        Returns:
            Vacuum status.
        """
        return self._run_tool("pg_vacuum", {"table": table})

    def pg_show_tables(self) -> str:
        """List PostgreSQL tables.

        Returns:
            Table list.
        """
        return self._run_tool("pg_show_tables", {})

    # --- Redis ---
    def redis_info(self) -> str:
        """Get Redis INFO diagnostics.

        Returns:
            Redis INFO output.
        """
        return self._run_tool("redis_info", {})

    def redis_slowlog(self) -> str:
        """Inspect Redis slowlog entries.

        Returns:
            Slowlog output.
        """
        return self._run_tool("redis_slowlog", {})

    def redis_keys(self, pattern: str = "*") -> str:
        """List Redis keys by pattern.

        Args:
            pattern: Redis key pattern.

        Returns:
            Matching keys.
        """
        return self._run_tool("redis_keys", {"pattern": pattern})

    def redis_flush_db(self) -> str:
        """Flush Redis DB.

        Returns:
            Flush result.
        """
        return self._run_tool("redis_flush_db", {})

    def redis_get_key(self, key: str) -> str:
        """Get value of a Redis key.

        Args:
            key: Redis key.

        Returns:
            Key value.
        """
        return self._run_tool("redis_get_key", {"key": key})

    # --- Infrastructure ---
    def docker_ps(self) -> str:
        """List Docker containers.

        Returns:
            Container list.
        """
        return self._run_tool("docker_ps", {})

    def docker_stats(self, container: str) -> str:
        """Get Docker resource stats for a container.

        Args:
            container: Container name.

        Returns:
            Stats output.
        """
        return self._run_tool("docker_stats", {"container": container})

    def docker_restart(self, container: str) -> str:
        """Restart a container.

        Args:
            container: Container name.

        Returns:
            Restart result.
        """
        return self._run_tool("docker_restart", {"container": container})

    def docker_logs(self, container: str) -> str:
        """Read logs for a container.

        Args:
            container: Container name.

        Returns:
            Container logs.
        """
        return self._run_tool("docker_logs", {"container": container})

    def check_disk_usage(self) -> str:
        """Check disk usage on host/container runtime.

        Returns:
            Disk usage summary.
        """
        return self._run_tool("check_disk_usage", {})

    # --- Resolution ---
    def diagnose_root_cause(self, root_cause: str) -> str:
        """Record a root-cause diagnosis.

        Args:
            root_cause: One-sentence root-cause summary.

        Returns:
            Acknowledgement from environment.
        """
        return self._run_tool("diagnose_root_cause", {"root_cause": root_cause})

    def write_postmortem(self, root_cause: str, impact: str, fix_applied: str, prevention: str) -> str:
        """Record a comprehensive post-mortem.

        Args:
            root_cause: The underlying cause.
            impact: Scope of disruption.
            fix_applied: Steps taken to fix.
            prevention: How to prevent.

        Returns:
            Acknowledgement from environment.
        """
        return self._run_tool("write_postmortem", {
            "root_cause": root_cause,
            "impact": impact,
            "fix_applied": fix_applied,
            "prevention": prevention,
        })

    def done(self) -> str:
        """Mark incident handling as complete (only after the stack is fixed).

        Returns:
            Final environment message.
        """
        return self._run_tool("done", {})


# =============================================================================
# Reward functions for GRPOTrainer
# =============================================================================

def make_reward_total(no_tool_penalty: float = -0.5):
    """Return ``reward_total`` closure used as the single primary reward.

    If the agent's completion did not parse to any tool call, the env was
    never stepped → ``len(env.trajectory) == 0``. We return ``no_tool_penalty``
    instead of 0.0 so format failures get a real gradient (matches
    kube-sre-gym's ``-0.5`` for unparsed completions).
    """

    def reward_total(completions=None, environments=None, **kwargs) -> List[float]:
        if not environments:
            count = len(completions) if completions is not None else 0
            return [0.0 for _ in range(count)]
        values: List[float] = []
        for env in environments:
            if not env.trajectory:
                values.append(float(no_tool_penalty))
            else:
                values.append(float(env.total_reward))
            try:
                env.maybe_log_and_record()
            except Exception as e:
                logger.warning(f"reward_total: episode logging failed: {e}")
        return values

    return reward_total


def reward_diagnosis_metric(completions=None, environments=None, **kwargs) -> List[float]:
    """Logging-only metric — kept out of ``reward_funcs`` so TRL doesn't sum it."""
    if not environments:
        return [0.0 for _ in range(len(completions) if completions else 0)]
    return [float(getattr(env, "diagnosis_reward", 0.0)) for env in environments]


def reward_fix_metric(completions=None, environments=None, **kwargs) -> List[float]:
    if not environments:
        return [0.0 for _ in range(len(completions) if completions else 0)]
    return [float(getattr(env, "fix_reward", 0.0)) for env in environments]


# =============================================================================
# Plotting
# =============================================================================

def sanitize_name(name: str) -> str:
    return name.replace("/", "-")


def plot_rewards(csv_path: Path, out_path: Path | None = None) -> None:
    """Plot reward curves from the CSV log."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    episodes, totals, fixes, diags = [], [], [], []
    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader, None)
        col = {c: i for i, c in enumerate(header or [])}
        for row in reader:
            if not row:
                continue
            try:
                episodes.append(int(float(row[col["episode"]])))
                totals.append(float(row[col["total_reward"]]))
                diags.append(float(row[col["diagnosis_reward"]]))
                fixes.append(float(row[col["fix_reward"]]))
            except Exception:
                continue

    if not episodes:
        logger.warning("No episodes to plot")
        return

    window = min(10, len(episodes))

    def rolling(vals: List[float]) -> List[float]:
        return [sum(vals[max(0, i - window):i + 1]) / min(i + 1, window) for i in range(len(vals))]

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.plot(episodes, totals, alpha=0.25, color="blue", marker="o", markersize=3, label="Per episode")
    ax.plot(episodes, rolling(totals), color="blue", linewidth=2.5, label=f"Rolling avg ({window})")
    z = np.polyfit(episodes, totals, 1)
    trend = np.poly1d(z)
    ax.plot(episodes, trend(episodes), color="red", linewidth=1.5, linestyle="--",
            label=f"Trend ({'↑' if z[0] > 0 else '↓'} {abs(z[0]):.3f}/ep)")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("PageZero SRE Agent — GRPO Training Reward Curve")
    ax.grid(True, alpha=0.3)
    ax.legend()

    save_path = out_path or csv_path.with_suffix(".png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Reward plot saved to {save_path}")


# =============================================================================
# CLI entry point
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO training for PageZero SRE agent")
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--env-url", default="http://localhost:8000")
    parser.add_argument("--dataset-size", type=int, default=30)
    parser.add_argument("--max-turns", type=int, default=8,
                        help="Max tool-calling iterations per episode (memory recipe: 6-8)")
    parser.add_argument("--max-new-tokens", type=int, default=1024,
                        help="max_completion_length per turn (Tier-1: 1024 for multi-step tool IO)")
    parser.add_argument("--num-generations", type=int, default=4,
                        help="G for GRPO (memory recipe: 4-6 on T4)")
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--save-steps", type=int, default=10)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-repo", default=None)
    parser.add_argument("--vllm-mode", choices=("colocate", "server"), default="colocate")
    parser.add_argument("--vllm-server-url", default="http://localhost:8001")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Sampling temperature (Tier-1: 0.6 reduces diagnose_root_cause mode-collapse)")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Nucleus sampling (set on GRPOConfig when supported)")
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--report-to", default="none", choices=("tensorboard", "wandb", "none"))
    parser.add_argument("--reward-log", default="reward_log.csv",
                        help="Top-level CSV for per-episode reward logging")
    parser.add_argument("--curriculum", action="store_true",
                        help="Run easy → medium → hard with mastery-gated advancement")
    return parser.parse_args()


def main() -> None:
    patch_trl_vllm_compat()
    args = parse_args()

    logger.info("=" * 60)
    logger.info("PageZero SRE Agent — GRPO Training (OpenEnv + TRL)")
    logger.info("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    PageZeroToolEnv.BASE_URL = args.env_url

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    default_output_dir = Path("outputs") / f"pagezero-sre-grpo-{sanitize_name(args.model_id)}-{timestamp}"
    output_dir = Path(args.output_dir or default_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Top-level reward log + matplotlib plot at output_dir root.
    reward_logger = RewardLogger(output_dir, csv_name=args.reward_log)
    PageZeroToolEnv.REWARD_LOGGER = reward_logger

    # Mastery-gated curriculum (kube-sre-gym style).
    stage_pools = {
        "easy":   ["task_1", "task_2", "task_3", "task_4", "task_5"],
        "medium": ["task_6", "task_7", "task_8", "task_9", "task_10"],
        "hard":   ["task_11", "task_12"],
    }
    PageZeroToolEnv.CURRICULUM = MasteryCurriculum(stage_task_pools=stage_pools)

    # One row per episode: prompt carries EPISODE_BRIEF; ``episode_task_id`` is
    # threaded into ``PageZeroToolEnv.reset`` by TRL so env == user assignment.
    dataset = build_grpo_dataset(
        args.dataset_size, stage_pools["easy"], group_size=args.num_generations
    )
    if args.dataset_size < args.num_generations:
        raise ValueError(
            "dataset_size must be >= num_generations for same-task grouping; "
            f"got dataset_size={args.dataset_size}, num_generations={args.num_generations}"
        )
    if args.dataset_size % args.num_generations != 0:
        raise ValueError(
            "dataset_size must be divisible by num_generations so each group is full; "
            f"got dataset_size={args.dataset_size}, num_generations={args.num_generations}"
        )

    grpo_fields = set(getattr(GRPOConfig, "__dataclass_fields__", {}).keys())
    grpo_kwargs: Dict[str, Any] = dict(
        use_vllm=True,
        vllm_mode=args.vllm_mode,
        vllm_server_base_url=args.vllm_server_url if args.vllm_mode == "server" else None,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        output_dir=str(output_dir),
        max_steps=args.max_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=2,
        max_grad_norm=1.0,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_train_batch_size=1,
        generation_batch_size=args.num_generations,
        num_generations=args.num_generations,
        remove_unused_columns=False,
        shuffle_dataset=False,
        max_completion_length=args.max_new_tokens,
        max_tool_calling_iterations=args.max_turns,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        temperature=args.temperature,
        report_to=args.report_to,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_repo if args.push_to_hub else None,
        hub_strategy="every_save",
        save_total_limit=3,
        loss_type="dapo",
        mask_truncated_completions=False,
        beta=0.01,
        chat_template_kwargs={"enable_thinking": False},
    )
    if "top_p" in grpo_fields:
        grpo_kwargs["top_p"] = args.top_p
    grpo_config = GRPOConfig(**grpo_kwargs)
    logger.info(
        "[preflight] grouping guard: remove_unused_columns=%s shuffle_dataset=%s "
        "dataset_size=%s g=%s",
        getattr(grpo_config, "remove_unused_columns", None),
        getattr(grpo_config, "shuffle_dataset", None),
        args.dataset_size,
        args.num_generations,
    )

    try:
        from peft import LoraConfig
    except Exception as e:
        raise ImportError(
            "peft is required to run train.py with LoRA. "
            "Install a compatible peft/transformers combination."
        ) from e

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    reward_total = make_reward_total(no_tool_penalty=PageZeroToolEnv.NO_TOOL_PENALTY)

    trainer = GRPOTrainer(
        model=args.model_id,
        processing_class=tokenizer,
        # Single primary reward: do NOT pass diagnosis/fix here — they would
        # be summed by TRL's default reward aggregator and double-count.
        reward_funcs=[reward_total],
        train_dataset=dataset,
        args=grpo_config,
        peft_config=peft_config,
        environment_factory=PageZeroToolEnv,
    )

    logger.info(f"Starting GRPO training (env={args.env_url}, G={args.num_generations})")
    try:
        if args.curriculum:
            for stage_name, task_ids in stage_pools.items():
                PageZeroToolEnv._set_stage(stage_name, task_ids)
                trainer.train_dataset = build_grpo_dataset(
                    args.dataset_size, task_ids, group_size=args.num_generations
                )
                logger.info(f"=== STAGE: {stage_name} (tasks={task_ids}) ===")
                trainer.train()
        else:
            PageZeroToolEnv._set_stage("easy", stage_pools["easy"])
            trainer.train()
    finally:
        PageZeroToolEnv._close_all()

    try:
        plot_rewards(reward_logger.csv_path, output_dir / "reward_plot.png")
    except Exception as e:
        logger.warning(f"Could not generate reward plot: {e}")

    trainer.save_model(str(output_dir))
    logger.info(f"Model saved to {output_dir}")
    logger.info(f"Reward log: {reward_logger.csv_path}")

    if args.push_to_hub and args.hub_repo:
        trainer.push_to_hub()
        logger.info(f"Model pushed to https://huggingface.co/{args.hub_repo}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
