import json
import os
from pydantic import BaseModel, Field

from dotenv import load_dotenv

from .config import (
    REWARD_TRIAGE_FIRST, REWARD_WRONG_FIRST,
    REWARD_CORRECT_ORDER, REWARD_SKIPPED_PHASE, REWARD_BACKWARD_PHASE,
    TERMINAL_BASE_SCORE, TERMINAL_HEALTHY_BONUS, TERMINAL_UNHEALTHY_PENALTY,
    TERMINAL_UNNECESSARY_FLUSH, TERMINAL_RECKLESS_RESTART,
    TERMINAL_ROOT_CAUSE_BONUS, TERMINAL_SLA_VIOLATED,
    TERMINAL_EXPECTED_FIX_BONUS, TERMINAL_WRONG_FIX_PENALTY,
    TERMINAL_RECKLESS_THRESHOLD,
    TERMINAL_ERROR_PENALTY_PER, TERMINAL_REPEAT_PENALTY_PER,
    TERMINAL_EFFICIENCY_WEIGHT,
    DEFAULT_MAX_STEPS,
)

load_dotenv()


class StepEvaluation(BaseModel):
    """Structured output for per-step reward scoring."""
    reward: float = Field(
        ..., ge=-0.3, le=0.2,
        description="Per-step reward. Positive for correct SRE actions, negative for harmful/wasteful ones."
    )
    rationale: str = Field(..., description="One-sentence explanation of the reward.")


class EvaluationOutput(BaseModel):
    """Structured output schema for terminal state evaluation."""
    score: float = Field(..., ge=0.0, le=1.0, description="Performance score from 0.0 (terrible) to 1.0 (perfect)")
    feedback: str = Field(..., description="Concise 1-2 sentence feedback explaining the score")

_TRIAGE_TOOLS = {"check_alerts", "get_service_metrics", "get_error_rate"}
_INVESTIGATE_TOOLS = {
    "pg_stat_activity", "pg_locks", "redis_info", "redis_slowlog",
    "read_app_logs", "search_logs", "docker_ps", "docker_stats",
    "pg_show_tables", "check_disk_usage", "redis_keys", "redis_get_key",
    "docker_logs", "curl_endpoint",
}
_DIAGNOSE_TOOLS = {"pg_explain_analyze", "pg_stat_statements", "get_recent_deploys"}
_FIX_TOOLS = {
    "pg_cancel_query", "pg_create_index", "pg_vacuum", "redis_flush_db",
    "docker_restart", "rollback_deploy",
}
_VERIFY_TOOLS = {"get_service_metrics", "curl_endpoint", "pg_stat_activity", "redis_info"}
_DOCUMENT_TOOLS = {"diagnose_root_cause", "write_postmortem", "done"}

PHASE_ORDER = {
    "triage": 0, "investigate": 1, "diagnose": 2,
    "fix": 3, "verify": 4, "document": 5,
}


def detect_phase(tool: str, history: list) -> str:
    """Map a tool to its SRE workflow phase.

    `history` is the *prior* history (not including the current tool). Triage
    tools are always classified as ``triage`` regardless of position; the
    "first-action triage bonus" is enforced in :func:`phase_score` instead so
    that workflow ordering after step 1 is still detected correctly.
    """
    has_fix = any(h.get("tool") in _FIX_TOOLS for h in history)
    if has_fix and tool in _VERIFY_TOOLS:
        return "verify"
    if tool in _TRIAGE_TOOLS:
        return "triage"
    if tool in _INVESTIGATE_TOOLS:
        return "investigate"
    if tool in _DIAGNOSE_TOOLS:
        return "diagnose"
    if tool in _FIX_TOOLS:
        return "fix"
    if tool in _DOCUMENT_TOOLS:
        return "document"
    return "investigate"


def phase_score(current_phase: str, history: list) -> float:
    """Reward correct SRE workflow ordering, penalize skipping or going backward.

    ``history`` may include the current step at index ``-1`` (that is the
    convention used by :class:`PageZeroEnvironment`). We always strip the last
    entry so ``past_phases`` represents only the *prior* trajectory; this also
    makes the "first action" branch reachable on step 1.
    """
    current_order = PHASE_ORDER.get(current_phase, 1)
    past_history = history[:-1] if history else []
    past_phases = [
        detect_phase(h.get("tool", ""), past_history[:i])
        for i, h in enumerate(past_history)
    ]

    if not past_phases:
        # First action: triage gets the bonus, anything else gets a penalty.
        return REWARD_TRIAGE_FIRST if current_phase == "triage" else REWARD_WRONG_FIRST

    max_past = max([PHASE_ORDER.get(p, 0) for p in past_phases] + [0])

    if current_order == max_past + 1:
        # Advanced exactly one phase — ideal
        return REWARD_CORRECT_ORDER
    elif current_order == max_past:
        # Stayed at the same phase (multiple investigate steps, etc.)
        return REWARD_CORRECT_ORDER * 0.5
    elif current_order > max_past + 1:
        # Skipped phases
        return REWARD_SKIPPED_PHASE
    else:
        # Went backward (e.g., Fix → Investigate)
        return REWARD_BACKWARD_PHASE


class LLMJudge:
    @staticmethod
    def _canonical_score(value: float) -> float:
        """Canonicalize score to strictly inside (0, 1) — never exactly 0.0 or 1.0.
        
        The OpenEnv validator rejects scores of exactly 0 or 1.
        This mirrors workspace_agent's DeterministicJudge._canonical_score().
        """
        clamped = max(0.01, min(0.99, float(value)))
        return round(clamped, 2)

    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if self.api_key and "your_" not in self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
            except ImportError:
                self.client = None
        else:
            self.client = None

    def get_phase_reward(self, tool: str, history: list,
                         scenario: dict | None = None) -> float:
        """Return a per-step reward using Gemini when available.

        Gemini judges whether the current tool is the right next action given
        the scenario context and recent history.  Falls back to deterministic
        :func:`phase_score` on any API error so the training signal is never
        a silent zero.
        """
        # ``history`` includes the current step at index -1 (env convention),
        # so build a *prior* view for the deterministic detector.
        prior_history = history[:-1] if history else []
        deterministic = phase_score(detect_phase(tool, prior_history), history)

        if not self.client or not scenario:
            return deterministic

        # The 'history' list now includes the current step at the very end.
        # We must split this into PRIOR HISTORY and the CURRENT ACTION to prevent overlap.
        prior_steps = history[:-1] if len(history) > 0 else []
        current_step = history[-1] if len(history) > 0 else {"tool": tool, "output": "None"}
        
        recent = prior_steps[-3:] if len(prior_steps) >= 3 else prior_steps
        recent_summary = []
        for h in recent:
            out = str(h.get("output", ""))
            had_error = out.startswith("ERROR")
            snippet = out[:200] + ("..." if len(out) > 200 else "")
            recent_summary.append({
                "tool": h["tool"],
                "output_snippet": f"[ERROR] {snippet}" if had_error else snippet,
                "had_error": had_error,
                "reward": h.get("reward", 0.0),
            })
            
        current_out = str(current_step.get("output", ""))
        current_had_error = current_out.startswith("ERROR")
        current_snippet = current_out[:400] + ("..." if len(current_out) > 400 else "")
        if current_had_error:
            current_snippet = f"[ERROR] {current_snippet}"

        prompt = f"""You are an expert SRE judge evaluating one step of an autonomous incident-response agent.

SCENARIO:
  Name: {scenario.get('name', 'unknown')}
  Alert: {scenario.get('alert', '')}
  Layer: {scenario.get('layer', '')}
  Root Cause: {scenario.get('root_cause', '')}
  Expected Fix: {', '.join(scenario.get('expected_fix', []))}

SRE WORKFLOW PHASES (in order):
  1. triage   — check_alerts, get_error_rate, get_service_metrics
  2. investigate — pg_stat_activity, pg_locks, redis_info, redis_slowlog, docker_ps, docker_logs, read_app_logs, search_logs, check_disk_usage, redis_keys
  3. diagnose — pg_explain_analyze, pg_stat_statements, get_recent_deploys
  4. fix      — pg_cancel_query, pg_create_index, pg_vacuum, redis_flush_db, docker_restart, rollback_deploy
  5. verify   — curl_endpoint, pg_stat_activity, redis_info, get_service_metrics
  6. document — diagnose_root_cause → write_postmortem → done

PRIOR HISTORY (last {len(recent_summary)} steps before now):
{json.dumps(recent_summary, indent=2)}

---
ACTION TO EVALUATE: {tool}
ACTION OUTCOME (What the tool returned): 
{current_snippet}
---

Rate this action with a reward in [-0.3, 0.2]:
  +0.20 = perfect next step for this phase/scenario
  +0.10 = reasonable but not ideal
   0.00 = neutral / no information gain
  -0.10 = wrong phase order OR irrelevant tool
  -0.20 = tool execution ERROR OR repeating a tool call OR reckless action
  -0.30 = destructive action OR hallucinating non-existent tool parameters

CRITICAL INSTRUCTIONS:
1. If the ACTION OUTCOME starts with [ERROR], you MUST provide a negative reward (e.g. -0.20). Do not reward "intent".
2. Explain your reasoning clearly based purely on the PRIOR HISTORY and ACTION OUTCOME. Do not invent history that didn't happen.

Return ONLY the JSON."""

        try:
            from google.genai import types
            
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",   # faster/cheaper model for per-step scoring
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_schema=StepEvaluation,
                    response_mime_type="application/json",
                )
            )
            data = json.loads(response.text)
            llm_reward = float(data.get("reward", 0.0))
            rationale = data.get("rationale", "")
            # Blend LLM signal with deterministic phase ordering so the reward
            # never collapses to 0 on a quirky LLM response and the workflow
            # bonus/penalty still pulls the policy toward correct ordering.
            blended = 0.7 * llm_reward + 0.3 * deterministic
            print(f"  [Judge] {tool}: LLM={llm_reward:+.2f} det={deterministic:+.2f}"
                  f" → {blended:+.2f} | {rationale}")
            return round(blended, 3)
        except Exception as e:
            # Rate limits / network errors → fall back to the deterministic
            # phase signal so the policy still gets a meaningful gradient.
            if "429" not in str(e) and "503" not in str(e):
                print(f"  [Judge] Step reward LLM error: {e} (using deterministic={deterministic:+.2f})")
            return deterministic

    def evaluate_terminal(
        self, scenario: dict, history: list, stack_healthy: bool, sla_status: dict
    ) -> tuple[float, float, str]:
        """Score a finished episode.

        Returns a 3-tuple ``(training_score, canonical_score, feedback)``:

        * ``training_score`` — uncapped value (typically in ``[-1.0, +1.0]``)
          consumed by the RL reward path. Negative values give the policy a
          real penalty for failure; previously this was clamped to
          ``[0.01, 0.99]`` and the failure gradient was lost.
        * ``canonical_score`` — clamped to ``(0, 1)`` so the OpenEnv validator
          (which rejects exact ``0`` / ``1``) is happy.
        * ``feedback`` — short human-readable rationale.
        """
        raw_score, fallback_feedback = self._fallback_evaluate(
            scenario, history, stack_healthy, sla_status
        )
        fallback_canonical = self._canonical_score(raw_score)

        if not self.client:
            return raw_score, fallback_canonical, fallback_feedback

        # Trim history output for the LLM prompt
        trimmed_history = []
        for h in history:
            trimmed = dict(h)
            out = str(trimmed.get("output", ""))
            if len(out) > 500:
                trimmed["output"] = out[:500] + "... [truncated]"
            trimmed_history.append(trimmed)

        prompt = f"""You are a Principal SRE Judge evaluating a junior agent's incident response.

SCENARIO:
Name: {scenario.get('name', 'unknown')}
Root Cause: {scenario.get('root_cause', 'unknown')}
Expected Fix Tools: {', '.join(scenario.get('expected_fix', []))}

OUTCOME:
Stack Health: {'HEALTHY (✓ Fixed)' if stack_healthy else 'UNHEALTHY (✗ Not Fixed)'}
SLA Status: {sla_status}

AGENT TRAJECTORY:
{json.dumps(trimmed_history, indent=2)}

GRADING PROFILE (Weights and Thresholds):
{json.dumps(scenario.get('grader_profile', {}), indent=2)}

Evaluate the agent's incident response based on:
1. Did they use proper SRE workflow (Triage → Investigate → Diagnose → Fix → Verify)?
2. Did they successfully diagnose and mitigate the issue?
3. Did they avoid reckless or destructive actions (considering the reckless_threshold in the profile)?
4. Did they use the expected fix tools if needed?
5. How does their performance align with the specific weighting in the profile?

Provide a score (0.0 = terrible/destructive, 1.0 = perfect) and brief feedback. Ensure the score reflects the profile's weighting.
"""

        try:
            from google.genai import types
            
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_schema=EvaluationOutput,
                    response_mime_type="application/json",
                )
            )
            data = json.loads(response.text)
            llm_canonical = float(data.get("score", fallback_canonical))
            feedback = str(data.get("feedback", fallback_feedback))
            # Map LLM canonical [0,1] → training scale [-1,+1] so failures
            # carry a real negative gradient. Then blend with the deterministic
            # raw score (50/50) for stability under noisy LLM judging.
            llm_training = (llm_canonical - 0.5) * 2.0
            training = 0.5 * llm_training + 0.5 * raw_score
            canonical = self._canonical_score(llm_canonical)
            return training, canonical, feedback
        except Exception as e:
            print(f"Gemini judge failed: {e} (using deterministic={raw_score:+.3f})")
            return raw_score, fallback_canonical, fallback_feedback

    def _fallback_evaluate(
        self, scenario: dict, history: list, stack_healthy: bool, sla_status: dict
    ) -> tuple[float, str]:
        """Programmatic evaluation with efficiency, error, and repetition penalties.

        Returns ``(raw_score, feedback)``. The raw score is intentionally
        *uncapped* — it is used as the training signal so that a complete
        failure produces a negative reward and a good run a positive one.
        :meth:`_canonical_score` is only applied later when reporting to the
        OpenEnv validator.
        """
        tools_used = [h["tool"] for h in history]
        num_steps = len(history)
        max_steps = DEFAULT_MAX_STEPS

        # Load per-task grading profile (unique weights per scenario)
        profile = scenario.get("grader_profile", {})
        healthy_bonus = profile.get("terminal_healthy_bonus", TERMINAL_HEALTHY_BONUS)
        unhealthy_penalty = profile.get("terminal_unhealthy_penalty", abs(TERMINAL_UNHEALTHY_PENALTY))
        reckless_threshold = profile.get("reckless_threshold", TERMINAL_RECKLESS_THRESHOLD)
        sla_penalty = profile.get("sla_penalty", abs(TERMINAL_SLA_VIOLATED))
        root_cause_bonus = profile.get("root_cause_bonus", TERMINAL_ROOT_CAUSE_BONUS)

        score = TERMINAL_BASE_SCORE
        feedback_parts = []

        # ── 1. Did they actually fix it? ──
        if stack_healthy:
            score += healthy_bonus
            feedback_parts.append("Restored cluster health.")
        else:
            score -= unhealthy_penalty
            feedback_parts.append("Cluster remained unhealthy.")

        # ── 2. Did they use the expected fix tools? ──
        expected_fixes = scenario.get("expected_fix", [])
        if expected_fixes:
            used_correct = [f for f in expected_fixes if f in tools_used]
            if used_correct:
                score += TERMINAL_EXPECTED_FIX_BONUS
                feedback_parts.append(f"Correctly used {', '.join(used_correct)}.")
            else:
                score += TERMINAL_WRONG_FIX_PENALTY
                feedback_parts.append(f"Did not use expected fix ({', '.join(expected_fixes)}).")

        # ── 3. Reckless actions ──
        if "redis_flush_db" in tools_used and scenario.get("layer") != "cache":
            score += TERMINAL_UNNECESSARY_FLUSH
            feedback_parts.append("Unnecessarily flushed Redis.")
        if "docker_restart" in tools_used:
            if num_steps < reckless_threshold:
                score += TERMINAL_RECKLESS_RESTART
                feedback_parts.append("Reckless restart without investigation.")

        # ── 4. Root cause bonus ──
        for h in history:
            if h["tool"] in ["diagnose_root_cause", "write_postmortem"]:
                score += root_cause_bonus
                feedback_parts.append("Good logging of root cause.")
                break

        # ── 5. SLA Penalty ──
        if sla_status.get("sla_status") == "VIOLATED":
            score -= sla_penalty
            feedback_parts.append("SLA violated due to slow response.")

        # ── 6. ERROR Penalty: penalize each step that returned an error ──
        error_count = sum(
            1 for h in history
            if str(h.get("output", "")).startswith("ERROR")
        )
        if error_count > 0:
            error_penalty = error_count * TERMINAL_ERROR_PENALTY_PER
            score += error_penalty
            feedback_parts.append(f"{error_count} tool errors ({error_penalty:+.2f}).")

        # ── 7. REPETITION Penalty: penalize redundant consecutive same-tool calls ──
        repeat_count = 0
        for i in range(1, len(history)):
            if history[i]["tool"] == history[i - 1]["tool"]:
                repeat_count += 1
        if repeat_count > 0:
            repeat_penalty = repeat_count * TERMINAL_REPEAT_PENALTY_PER
            score += repeat_penalty
            feedback_parts.append(f"{repeat_count} redundant repeats ({repeat_penalty:+.2f}).")

        # ── 8. EFFICIENCY Bonus: reward solving it quickly ──
        if num_steps > 0 and stack_healthy:
            efficiency_ratio = 1.0 - (num_steps / max_steps)
            efficiency_bonus = TERMINAL_EFFICIENCY_WEIGHT * max(0.0, efficiency_ratio)
            score += efficiency_bonus
            feedback_parts.append(f"Efficiency: {num_steps}/{max_steps} steps ({efficiency_bonus:+.2f}).")

        profile_id = profile.get('profile_id', 'default')
        feedback = f"[grader={profile_id}] " + " ".join(feedback_parts)
        # Return the *raw* score (no canonical clamp here). The caller is
        # responsible for canonicalizing it for the validator if needed.
        return float(score), feedback

