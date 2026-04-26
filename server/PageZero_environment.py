import json
import logging
import re
import subprocess
import time
from typing import Any, Dict, Optional

from openenv.core.env_server.interfaces import Environment
from models import PageZeroAction, PageZeroObservation, PageZeroState
from uuid import uuid4
import os

from .stack_backend import StackBackend
from .executor import Executor
from .curriculum import Curriculum
from .llm_designer import LLMDesigner
from .llm_judge import (
    LLMJudge,
    FIX_TOOLS,
    detect_phase,
    persona_for_difficulty,
)
from .schema_drift import SchemaDriftEngine
from .config import (
    DEFAULT_MAX_STEPS,
    INJECTION_WAIT_SECONDS,
    MIN_STEPS_BEFORE_DONE,
    MIN_STEPS_BEFORE_RESOLVE,
    REQUIRE_DOCS_BEFORE_SUCCESS,
    REWARD_DONE_BEFORE_DOCS,
    REWARD_DONE_UNRESOLVED,
    REWARD_EARLY_DOC_BLOCK,
    REWARD_REPEAT_2X,
    REWARD_REPEAT_3X,
    TERMINAL_RESOLVED_BASE,
    TERMINAL_RESOLVED_DIFF_WEIGHT,
    TERMINAL_RESOLVED_EFFICIENCY_WEIGHT,
    TERMINAL_TIMEOUT_FAIL_TOTAL,
)

logger = logging.getLogger(__name__)


# Tools that should *always* trigger the two-stage resolution gate even if
# the agent did not call them with a "fix:" prefix. We treat anything in the
# canonical FIX_TOOLS set as a mutation; specific tool names (e.g.
# "redis_flush_db", "docker_restart") are also mutations.
_AUTO_FIX_TOOLS = set(FIX_TOOLS)


def _normalize_args(args: Dict[str, Any]) -> str:
    """Produce a deterministic string for repeat-detection."""
    try:
        return json.dumps(args or {}, sort_keys=True, default=str)
    except Exception:
        return str(args or {})


class PageZeroEnvironment(Environment):
    """PageZero Gym Environment for OpenEnv.

    Each episode:
      1. Cleans up residual state from the previous episode
      2. Picks a scenario matched to the current difficulty
      3. Injects faults into the live Docker stack
      4. Lets the agent diagnose and fix the incident
      5. Scores the agent and feeds the result back into the curriculum

    Reward shape (kube-sre-gym aligned):
      * Per-step:    deterministic phase shaping + LLM judge blend (in [-0.3, +0.2])
      * Repeat:      -0.30 on 2nd identical call, -0.50 + circuit breaker on 3rd
      * Parse fail:  caller (notebook / train.py) is responsible for -0.5 on no-tool
      * Terminal:    on confirmed resolution, +base + difficulty*W + (1 - steps/max)*W
                     on timeout/unresolved-done, total reward is wiped to ``-2.0``
    """
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.backend = StackBackend()
        self.executor = Executor(self.backend)
        self.curriculum = Curriculum()
        self.designer = LLMDesigner()
        self.judge = LLMJudge()
        self.drift_engine = SchemaDriftEngine()

        self._step_count = 0
        self._history: list[dict] = []
        self._scenario: Optional[dict] = None
        self._max_steps = DEFAULT_MAX_STEPS
        self._episode_count = 0
        self._cumulative_reward = 0.0
        self._call_counts: Dict[str, int] = {}
        self._state = PageZeroState(episode_id=str(uuid4()), step_count=0)

    # ─────────────────────────────────────────────────────────────────────
    # Episode lifecycle
    # ─────────────────────────────────────────────────────────────────────
    def reset(self, task_id: Optional[str] = None, **kwargs) -> PageZeroObservation:
        """Reset the environment to a new chaotic state."""
        self._step_count = 0
        self._history = []
        self._episode_count += 1
        self._cumulative_reward = 0.0
        self._call_counts = {}
        self._used_diagnose_root_cause = False
        self._used_write_postmortem = False

        self._cleanup_previous_episode()

        task_id = task_id or os.environ.get("TASK_ID")

        if task_id:
            scenario = self.designer.get_scenario_by_id(task_id)
            difficulty = scenario.get("difficulty", 0.5)
            use_warmup = False
            weakest_layer = scenario.get("layer", "")
            logger.info(f"Validator Override: Testing specific task {task_id}")
        else:
            difficulty = self.curriculum.get_difficulty()
            use_warmup = self.curriculum.should_use_warmup()
            weakest_layer = self.curriculum.get_weakest_layer()

            scenario = self.designer.design(
                self.curriculum.skill_profile,
                difficulty,
                use_warmup=use_warmup,
                weakest_layer=weakest_layer,
            )

        self._scenario = scenario
        logger.info(
            f"Episode {self._episode_count}: "
            f"difficulty={difficulty:.2f}, "
            f"scenario={scenario['name']} ({scenario.get('task_id', 'LLM')}), "
            f"layer={scenario.get('layer', '?')}"
        )

        self._state = PageZeroState(
            episode_id=str(uuid4()),
            step_count=0,
            difficulty=difficulty,
            scenario_name=self._scenario.get("name") if self._scenario else "None",
            is_resolved=False,
            cumulative_reward=0.0,
        )

        for cmd in scenario.get("inject_commands", []):
            try:
                subprocess.run(cmd, shell=True, timeout=30,
                               capture_output=True, text=True)
            except subprocess.TimeoutExpired:
                logger.debug(f"Injection command timed out (expected for bg): {cmd[:80]}")
            except Exception as e:
                logger.error(f"Failed to run injection command {cmd}: {e}")

        time.sleep(INJECTION_WAIT_SECONDS)
        self.backend.reset_incident_timer()

        return PageZeroObservation(
            tool_output=f"🚨 ALERT: {scenario.get('alert', 'Unknown Anomaly')}",
            active_alerts=[scenario.get("alert", "")],
            sla_status="OK",
            revenue_loss_usd=0.0,
            downtime_minutes=0.0,
            step=0,
            max_steps=self._max_steps,
            phase_history=[],
            stack_healthy=None,
            judge_feedback=None,
            phase=None,
            is_fix_step=False,
            repeat_count=1,
        )

    def _cleanup_previous_episode(self):
        """Undo all side effects from the previous episode so we start clean."""
        try:
            self.backend.cleanup_postgres()
            self.backend.cleanup_redis()
            self.backend.revert_schema_drift()
            self.drift_engine.reset()
            time.sleep(1)
        except Exception as e:
            logger.warning(f"Cleanup failed (non-fatal): {e}")

    # ─────────────────────────────────────────────────────────────────────
    # Step
    # ─────────────────────────────────────────────────────────────────────
    def step(self, action: PageZeroAction) -> PageZeroObservation:
        """Execute a tool, evaluate it, and return a fully-populated obs."""
        self._step_count += 1
        self._state.step_count = self._step_count

        tool = action.tool
        args = action.args or {}
        call_key = f"{tool}::{_normalize_args(args)}"
        repeat_count = self._call_counts.get(call_key, 0) + 1
        self._call_counts[call_key] = repeat_count

        # 1) Schema drift may inject fresh failures mid-episode.
        self._maybe_apply_drift()

        # 2) Tier-1 guard: documentation tools only after triage/investigate work.
        early_doc_block = False
        circuit_broken = False
        if tool in ("diagnose_root_cause", "write_postmortem") and self._step_count < 3:
            early_doc_block = True
            output = (
                "BLOCKED: `diagnose_root_cause` and `write_postmortem` are documentation "
                "steps and are only allowed starting at environment step 3 (after at least "
                "two prior tool calls). Use triage/investigate tools first (e.g. "
                "`check_alerts`, `pg_stat_activity`, `redis_info`)."
            )
        # 3) Repeat-command circuit breaker: 3rd identical call is blocked.
        elif repeat_count >= 3:
            output = (
                f"BLOCKED: You already tried `{tool}` with these arguments "
                f"{repeat_count} times. Try a different approach or different arguments."
            )
            circuit_broken = True
        else:
            output = self.executor.execute(tool, args)
            if tool == "diagnose_root_cause":
                self._used_diagnose_root_cause = True
            elif tool == "write_postmortem":
                self._used_write_postmortem = True

        # Track history before judging so the judge sees this step.
        self._history.append({
            "tool": tool,
            "args": args,
            "output": output,
            "reward": 0.0,  # placeholder
        })

        # 4) Per-step reward.
        if circuit_broken:
            step_reward = REWARD_REPEAT_3X
            judge_feedback = "Command blocked — repeated too many times."
        elif early_doc_block:
            step_reward = float(REWARD_EARLY_DOC_BLOCK)
            judge_feedback = (
                "Blocked: documentation tools are not allowed before step 3 — "
                "run triage/investigate tools first."
            )
        else:
            step_reward, judge_feedback = self.judge.get_phase_reward(
                tool, self._history, scenario=self._scenario, return_feedback=True
            )
            step_reward = float(step_reward or 0.0)

            if str(output).startswith("ERROR"):
                step_reward -= 0.15
                judge_feedback = (judge_feedback or "") + " | ERROR penalty -0.15"
                logger.info("  [Enforcer] Applied -0.15 penalty for ERROR output")

            if repeat_count == 2:
                step_reward += REWARD_REPEAT_2X
                judge_feedback = (judge_feedback or "") + " | repeat 2x penalty"

        # Persist the per-step reward so the terminal reward decomposition
        # in the wrapper stays exact.
        self._history[-1]["reward"] = step_reward
        self._cumulative_reward += step_reward
        self._state.cumulative_reward = self._cumulative_reward

        # 5) Phase classification (for the obs and for downstream logging).
        prior_history = self._history[:-1]
        phase = detect_phase(tool, prior_history)

        # 6) Auto-detect fix tools (run resolution gate even without a prefix).
        is_fix_step = (tool in _AUTO_FIX_TOOLS)
        difficulty = self._state.difficulty
        persona = persona_for_difficulty(difficulty)

        sla_info = self.backend.get_sla_status()

        # 7) Two-stage resolution gate: programmatic poll + LLM verifier.
        gate_resolved = False
        gate_reason = ""
        stack_healthy: Optional[bool] = None
        if is_fix_step and not circuit_broken:
            stack_healthy = self.backend.verify_resolution()
            if stack_healthy:
                snapshot = self._snapshot_text()
                gate_resolved, gate_reason = self.judge.verify_resolution(
                    self._scenario or {}, self._history, snapshot
                )
                if gate_resolved:
                    judge_feedback = (
                        f"RESOLVED: programmatic + judge confirmed. {gate_reason}"
                    )
                else:
                    judge_feedback = (
                        f"Stack looks healthy but judge disagrees: {gate_reason}"
                    )
            else:
                judge_feedback = (judge_feedback or "") + " | stack still unhealthy"

        docs_complete = (
            self._used_diagnose_root_cause and self._used_write_postmortem
        )
        docs_required = bool(REQUIRE_DOCS_BEFORE_SUCCESS)
        docs_ready = (not docs_required) or docs_complete

        # 8) Termination logic: timeout OR (resolved + docs) OR accepted done.
        gate_resolve_ready = (
            gate_resolved and self._step_count >= int(MIN_STEPS_BEFORE_RESOLVE)
        )
        deferred_gate_resolve = (
            gate_resolved and self._step_count < int(MIN_STEPS_BEFORE_RESOLVE)
        )
        done_requested = (tool == "done")
        done_step_ready = (
            done_requested and self._step_count >= int(MIN_STEPS_BEFORE_DONE)
        )
        premature_done = (done_requested and not done_step_ready)
        unresolved_done = (done_step_ready and not gate_resolve_ready)
        docs_missing_done = (done_step_ready and gate_resolve_ready and not docs_ready)
        done_accepted = (done_step_ready and gate_resolve_ready and docs_ready)
        timeout = (self._step_count >= self._max_steps)
        done = timeout or done_accepted or (gate_resolve_ready and docs_ready)

        if premature_done or unresolved_done:
            step_reward += float(REWARD_DONE_UNRESOLVED)
        if docs_missing_done:
            step_reward += float(REWARD_DONE_BEFORE_DOCS)

        canonical_score: Optional[float] = None
        terminal_training_score = 0.0

        if done:
            # If the agent calls "done" without the gate confirming, do a
            # last-chance programmatic + judge check so we never label an
            # unfixed incident as resolved.
            if done_accepted and not gate_resolve_ready and not timeout:
                stack_healthy = self.backend.verify_resolution()
                if stack_healthy:
                    snapshot = self._snapshot_text()
                    gate_resolved, gate_reason = self.judge.verify_resolution(
                        self._scenario or {}, self._history, snapshot
                    )
                    gate_resolve_ready = (
                        gate_resolved and self._step_count >= int(MIN_STEPS_BEFORE_RESOLVE)
                    )

            if timeout and not (gate_resolve_ready and docs_ready):
                # kube-sre-gym wipe: net total reward becomes a clean -2.0
                raw_sum = sum(h["reward"] for h in self._history)
                terminal_training_score = TERMINAL_TIMEOUT_FAIL_TOTAL - raw_sum
                step_reward += terminal_training_score
                self._cumulative_reward = TERMINAL_TIMEOUT_FAIL_TOTAL
                self._state.cumulative_reward = self._cumulative_reward
                judge_feedback = "Timeout — incident remains unresolved (reward wiped to -2.0)."
                canonical_score = self.judge._canonical_score(0.0)
                self._state.is_resolved = False
            elif gate_resolve_ready and docs_ready:
                base_bonus = TERMINAL_RESOLVED_BASE + difficulty * TERMINAL_RESOLVED_DIFF_WEIGHT
                efficiency = TERMINAL_RESOLVED_EFFICIENCY_WEIGHT * (
                    1.0 - self._step_count / max(1, self._max_steps)
                )
                terminal_training_score = base_bonus + efficiency
                step_reward += terminal_training_score
                self._cumulative_reward += terminal_training_score
                self._state.cumulative_reward = self._cumulative_reward
                self._state.is_resolved = True
                # Canonical score uses the LLM-blended terminal grader so the
                # OpenEnv validator still sees a meaningful 0..1 number.
                _train, canonical_score, eval_feedback = self.judge.evaluate_terminal(
                    self._scenario or {}, self._history, True, sla_info, persona=persona
                )
                judge_feedback = (
                    f"RESOLVED (+{terminal_training_score:.2f}). {eval_feedback}"
                )
            else:
                # Agent called "done" but neither gate confirmed — penalize as
                # premature termination using the programmatic grader.
                training_score, canonical_score, eval_feedback = self.judge.evaluate_terminal(
                    self._scenario or {}, self._history,
                    bool(stack_healthy), sla_info, persona=persona,
                )
                terminal_training_score = float(training_score)
                step_reward += terminal_training_score
                self._cumulative_reward += terminal_training_score
                self._state.cumulative_reward = self._cumulative_reward
                self._state.is_resolved = bool(stack_healthy)
                judge_feedback = (
                    f"Premature done ({terminal_training_score:+.2f}). {eval_feedback}"
                )

            done_cause = (
                "gate_resolved" if (gate_resolve_ready and docs_ready)
                else "timeout" if timeout
                else "done_tool" if done_accepted
                else "unknown"
            )
            # Feed result back into the legacy curriculum (used by the
            # designer/weak-layer logic; the new mastery curriculum lives
            # client-side in the trainer).
            scenario_layer = (self._scenario or {}).get("layer", "cross_layer")
            self.curriculum.record_result(scenario_layer, canonical_score or 0.0)

            logger.info(
                f"Episode {self._episode_count} done: "
                f"resolved={self._state.is_resolved} "
                f"terminal_train={terminal_training_score:+.2f} "
                f"canonical={canonical_score} "
                f"steps={self._step_count}/{self._max_steps} "
                f"cause={done_cause} "
                f"docs_complete={docs_complete} "
                f"resolve_ready={gate_resolve_ready} "
                f"done_accepted={done_accepted}"
            )
        elif premature_done:
            judge_feedback = (
                f"Ignored `done` before min step {MIN_STEPS_BEFORE_DONE}; continue investigation. "
                f"penalty={REWARD_DONE_UNRESOLVED:+.2f}"
            )
        elif unresolved_done:
            judge_feedback = (
                "Ignored `done`: stack not resolved yet. "
                f"penalty={REWARD_DONE_UNRESOLVED:+.2f}"
            )
        elif docs_missing_done:
            judge_feedback = (
                "Resolved but documentation incomplete: call both "
                "`diagnose_root_cause` and `write_postmortem` before `done`. "
                f"penalty={REWARD_DONE_BEFORE_DOCS:+.2f}"
            )
        elif deferred_gate_resolve:
            judge_feedback = (
                f"Resolution confirmed but deferred until step >= {MIN_STEPS_BEFORE_RESOLVE} "
                f"for deeper trajectories."
            )

        # 9) Auto-include a fresh stack snapshot after a fix or on done so the
        #    agent does not need to spend a turn re-checking.
        if (is_fix_step or done) and not circuit_broken and not early_doc_block:
            try:
                snapshot = self._snapshot_text()
                output = f"{output}\n\n--- POST-ACTION STACK SNAPSHOT ---\n{snapshot}"
            except Exception:
                pass

        return PageZeroObservation(
            tool_output=output,
            active_alerts=[(self._scenario or {}).get("alert", "")] if not done else [],
            sla_status=sla_info["sla_status"],
            revenue_loss_usd=sla_info["revenue_loss_usd"],
            downtime_minutes=sla_info["downtime_minutes"],
            step=self._step_count,
            max_steps=self._max_steps,
            hint=judge_feedback,
            phase_history=[h["tool"] for h in self._history],
            is_done=done,
            stack_healthy=stack_healthy if stack_healthy is not None else (
                self._state.is_resolved if done else None
            ),
            final_score=canonical_score,
            reward=step_reward,
            judge_feedback=judge_feedback,
            phase=phase,
            is_fix_step=is_fix_step,
            repeat_count=repeat_count,
            done=done,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────
    def _maybe_apply_drift(self) -> None:
        drift = self.drift_engine.maybe_drift(
            self._step_count, self.curriculum.get_difficulty()
        )
        if not drift:
            return
        logger.info(f"Schema Drift triggered: {drift['description']}")
        try:
            if drift.get("layer") == "database":
                self.backend._run_psql(drift["command"])
            elif drift.get("layer") == "cache":
                precondition = drift.get("precondition_cmd")
                if precondition:
                    result = self.backend._run_redis(*precondition.split())
                    if result.strip() != "1":
                        logger.info(
                            f"Drift skipped — precondition not met: {precondition}"
                        )
                        return
                self.backend._run_redis(*drift["command"].split())
        except Exception as e:
            logger.warning(f"Drift apply failed (non-fatal): {e}")

    def _snapshot_text(self) -> str:
        """Cheap stack snapshot used by the resolution gate + auto-snapshot.

        We try alerts → SLA → backend health. Any individual call may fail
        (e.g. Redis timing out) — we just include what we have so the LLM
        can still reason about the rest.
        """
        parts: list[str] = []
        try:
            alerts = self.executor.execute("check_alerts", {})
            parts.append(f"check_alerts:\n{alerts[:600]}")
        except Exception:
            pass
        try:
            sla = self.backend.get_sla_status()
            parts.append(
                f"sla: status={sla.get('sla_status')}, "
                f"downtime_min={sla.get('downtime_minutes')}, "
                f"revenue_loss=${sla.get('revenue_loss_usd')}"
            )
        except Exception:
            pass
        try:
            healthy = self.backend.verify_resolution()
            parts.append(f"backend.verify_resolution: {healthy}")
        except Exception:
            pass
        return "\n".join(parts) or "(no snapshot available)"

    def get_reward(self, observation: PageZeroObservation, done: bool) -> float:
        """The reward is handled in step(); just return 0 here."""
        return 0.0

    def get_info(self, observation: PageZeroObservation, done: bool) -> Dict[str, Any]:
        return {
            "scenario": (self._scenario or {}).get("name", "None"),
            "step_count": self._step_count,
            "difficulty": self.curriculum.get_difficulty(),
            "episodes_completed": self.curriculum.episodes_completed,
            "stack_healthy": self._state.is_resolved if self._step_count > 0 else None,
        }

    @property
    def state(self) -> PageZeroState:
        return self._state


# Re-export so tests / other consumers can introspect the auto-detect set.
AUTO_FIX_TOOLS = frozenset(_AUTO_FIX_TOOLS)
