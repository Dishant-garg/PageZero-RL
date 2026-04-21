import logging
import subprocess
import time
from typing import Any, Dict

from openenv.core.env_server.environment import BaseEnvironment
from openenv.core.env_server.types import StepResult
from models import PageZeroAction, PageZeroObservation
from .stack_backend import StackBackend
from .executor import Executor
from .curriculum import Curriculum
from .llm_designer import LLMDesigner
from .llm_judge import LLMJudge
from .schema_drift import SchemaDriftEngine

logger = logging.getLogger(__name__)

class PageZeroEnvironment(BaseEnvironment[PageZeroAction, PageZeroObservation]):
    """PageZero Gym Environment for OpenEnv."""

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
        self._max_steps = 15

    def reset(self) -> PageZeroObservation:
        """Reset the environment to a new chaotic state."""
        self._step_count = 0
        self._history = []
        self.drift_engine.has_drifted = False
        
        # 1. Pick scenario
        difficulty = self.curriculum.get_difficulty()
        if self.curriculum.should_use_warmup():
            # For hackathon/simplicity, we just use random warmup directly
            scenario = self.designer.design(self.curriculum.skill_profile, difficulty)
        else:
            scenario = self.designer.design(self.curriculum.skill_profile, difficulty)
        
        self._scenario = scenario
        
        # 2. Reset containers to known clean state (if required)
        # self.backend.reset_containers()
        
        # 3. Inject faults
        logger.info(f"Injecting scenario: {scenario['name']}")
        for cmd in scenario.get("inject_commands", []):
            try:
                subprocess.run(cmd, shell=True, timeout=10)
            except Exception as e:
                logger.error(f"Failed to run injection command {cmd}: {e}")
                
        # 4. Wait for faults to cascade into symptoms
        time.sleep(2)
        
        self.backend.reset_incident_timer()

        return PageZeroObservation(
            tool_output=f"🚨 ALERT: {scenario.get('alert', 'Unknown Anomaly')}",
            active_alerts=[scenario.get("alert", "")],
            sla_status="OK",
            revenue_loss_usd=0.0,
            downtime_minutes=0.0,
            step=0,
            max_steps=self._max_steps,
            phase_history=[]
        )

    def step(self, action: PageZeroAction) -> StepResult[PageZeroObservation]:
        """Execute a tool, evaluate the SRE phase, and track SLA."""
        self._step_count += 1
        tool = action.tool
        args = action.args

        # 1. Schema Drift?
        drift = self.drift_engine.maybe_drift(self._step_count, self.curriculum.get_difficulty())
        if drift:
            logger.info(f"Schema Drift triggered: {drift['description']}")
            if drift.get("layer") == "database":
                self.backend._run_psql(drift["command"])
            elif drift.get("layer") == "cache":
                self.backend._run_redis(drift["command"])

        # 2. Execute Tool
        output = self.executor.execute(tool, args)

        # 3. Evaluate Step (Phase)
        step_reward = self.judge.get_phase_reward(tool, self._history)
        
        # Track history
        self._history.append({
            "tool": tool,
            "args": args,
            "output": output,
            "reward": step_reward
        })

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
            final_score=final_score
        )

        return StepResult(observation=obs, reward=step_reward, done=done)

    def get_reward(self, observation: PageZeroObservation, done: bool) -> float:
        """The reward is handled in step(), just return 0 here."""
        return 0.0

    def get_info(self, observation: PageZeroObservation, done: bool) -> Dict[str, Any]:
        """Additional debug info."""
        return {
            "scenario": self._scenario.get("name") if self._scenario else "None",
            "step_count": self._step_count
        }
