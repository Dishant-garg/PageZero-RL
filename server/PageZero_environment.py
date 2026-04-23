import logging
import subprocess
import time
from typing import Any, Dict

from openenv.core.env_server.interfaces import Environment
from openenv.core.client_types import StepResult
from models import PageZeroAction, PageZeroObservation, PageZeroState
from uuid import uuid4
from .stack_backend import StackBackend
from .executor import Executor
from .curriculum import Curriculum
from .llm_designer import LLMDesigner
from .llm_judge import LLMJudge
from .schema_drift import SchemaDriftEngine
from .config import DEFAULT_MAX_STEPS, INJECTION_WAIT_SECONDS
import os
from typing import Optional

logger = logging.getLogger(__name__)


class PageZeroEnvironment(Environment):
    """PageZero Gym Environment for OpenEnv.

    Each episode:
      1. Cleans up residual state from the previous episode
      2. Picks a scenario matched to the current difficulty
      3. Injects faults into the live Docker stack
      4. Lets the agent diagnose and fix the incident
      5. Scores the agent and feeds the result back into the curriculum
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.backend = StackBackend()
        self.executor = Executor(self.backend)
        self.curriculum = Curriculum()
        self.designer = LLMDesigner()
        self.judge = LLMJudge()
        self.drift_engine = SchemaDriftEngine()

        self._step_count = 0
        self._history = []
        self._scenario = None
        self._max_steps = DEFAULT_MAX_STEPS
        self._episode_count = 0
        self._cumulative_reward = 0.0
        self._state = PageZeroState(episode_id=str(uuid4()), step_count=0)

    def reset(self, task_id: Optional[str] = None, **kwargs) -> PageZeroObservation:
        """Reset the environment to a new chaotic state."""
        self._step_count = 0
        self._history = []
        self._episode_count += 1
        self._cumulative_reward = 0.0

        # ── 1. Clean up residual state from the previous episode ──
        self._cleanup_previous_episode()

        # ── 2. Pick scenario ──
        # Listen to task_id from arg or environment (set by OpenEnv validator)
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
            cumulative_reward=0.0
        )

        # ── 3. Inject faults ──
        for cmd in scenario.get("inject_commands", []):
            try:
                subprocess.run(cmd, shell=True, timeout=30,
                               capture_output=True, text=True)
            except subprocess.TimeoutExpired:
                # Background commands (&) may time out — that's expected
                logger.debug(f"Injection command timed out (expected for bg): {cmd[:80]}")
            except Exception as e:
                logger.error(f"Failed to run injection command {cmd}: {e}")

        # ── 4. Wait for faults to cascade into symptoms ──
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
        )

    def _cleanup_previous_episode(self):
        """Undo all side effects from the previous episode so we start clean.

        This is critical for RL training — without it, faults from episode N
        bleed into episode N+1 and corrupt the reward signal.
        """
        try:
            # 1. Kill all runaway queries and locks in Postgres
            self.backend.cleanup_postgres()

            # 2. Flush Redis and re-enable active expiry
            self.backend.cleanup_redis()

            # 3. Revert any schema drifts (column renames, etc.)
            self.backend.revert_schema_drift()

            # 4. Reset the drift engine's internal state
            self.drift_engine.reset()

            # 5. Brief pause for Postgres to stabilize after killing backends
            time.sleep(1)
        except Exception as e:
            logger.warning(f"Cleanup failed (non-fatal): {e}")

    def step(self, action: PageZeroAction) -> StepResult[PageZeroObservation]:
        """Execute a tool, evaluate the SRE phase, and track SLA."""
        self._step_count += 1
        self._state.step_count = self._step_count
        
        tool = action.tool
        args = action.args

        # 1. Schema Drift?
        drift = self.drift_engine.maybe_drift(
            self._step_count, self.curriculum.get_difficulty()
        )
        if drift:
            logger.info(f"Schema Drift triggered: {drift['description']}")
            if drift.get("layer") == "database":
                self.backend._run_psql(drift["command"])
            elif drift.get("layer") == "cache":
                # Check precondition if present (e.g., key must exist)
                precondition = drift.get("precondition_cmd")
                if precondition:
                    result = self.backend._run_redis(*precondition.split())
                    # Redis EXISTS returns "1" if key exists
                    if result.strip() != "1":
                        logger.info(f"Drift skipped — precondition not met: {precondition}")
                    else:
                        self.backend._run_redis(*drift["command"].split())
                else:
                    self.backend._run_redis(*drift["command"].split())

        # 2. Execute Tool
        output = self.executor.execute(tool, args)

        # 3. Track history FIRST so the Judge can see this step's output
        self._history.append({
            "tool": tool,
            "args": args,
            "output": output,
            "reward": 0.0,  # placeholder, updated below
        })

        # 4. Evaluate Step (Phase) — Judge now sees full history including this step
        step_reward = self.judge.get_phase_reward(tool, self._history, scenario=self._scenario)
        
        # 4b. Deterministic Enforcement: Penalize Errors (Missing args, etc)
        if output.startswith("ERROR"):
            step_reward -= 0.15  # Hard penalty for invalid tool usage
            logger.info(f"  [Enforcer] Applied -0.15 penalty for ERROR output")

        # Update the placeholder reward in history
        self._history[-1]["reward"] = step_reward

        self._cumulative_reward += step_reward
        self._state.cumulative_reward = self._cumulative_reward

        # SLA Status
        sla_info = self.backend.get_sla_status()

        done = (tool == "done") or (self._step_count >= self._max_steps)
        final_score = None
        hint = None

        if done:
            # 4. Terminal Evaluation
            stack_healthy = self.backend.verify_resolution()
            final_score, hint = self.judge.evaluate_terminal(
                self._scenario, self._history, stack_healthy, sla_info
            )
            # Override reward to provide strong terminal signal
            step_reward += final_score
            self._cumulative_reward += final_score
            self._state.cumulative_reward = self._cumulative_reward
            self._state.is_resolved = stack_healthy

            # 5. Feed result back into the curriculum for difficulty progression
            scenario_layer = self._scenario.get("layer", "cross_layer")
            self.curriculum.record_result(scenario_layer, final_score)

            logger.info(
                f"Episode {self._episode_count} done: "
                f"score={final_score:.2f}, "
                f"healthy={stack_healthy}, "
                f"new_difficulty={self.curriculum.get_difficulty():.2f}"
            )

        obs = PageZeroObservation(
            tool_output=output,
            active_alerts=[self._scenario.get("alert", "")] if not done else [],
            sla_status=sla_info["sla_status"],
            revenue_loss_usd=sla_info["revenue_loss_usd"],
            downtime_minutes=sla_info["downtime_minutes"],
            step=self._step_count,
            max_steps=self._max_steps,
            hint=hint,
            phase_history=[h["tool"] for h in self._history],
            is_done=done,
            final_score=final_score,
        )

        return StepResult(observation=obs, reward=step_reward, done=done)

    def get_reward(self, observation: PageZeroObservation, done: bool) -> float:
        """The reward is handled in step(), just return 0 here."""
        return 0.0

    def get_info(self, observation: PageZeroObservation, done: bool) -> Dict[str, Any]:
        """Additional debug info."""
        return {
            "scenario": self._scenario.get("name") if self._scenario else "None",
            "step_count": self._step_count,
            "difficulty": self.curriculum.get_difficulty(),
            "episodes_completed": self.curriculum.episodes_completed,
        }

    @property
    def state(self) -> PageZeroState:
        return self._state
