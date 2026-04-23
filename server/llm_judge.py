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
_DOCUMENT_TOOLS = {"diagnose_root_cause"}

PHASE_ORDER = {
    "triage": 0, "investigate": 1, "diagnose": 2,
    "fix": 3, "verify": 4, "document": 5,
}


def detect_phase(tool: str, history: list) -> str:
    has_fix = any(h.get("tool") in _FIX_TOOLS for h in history)
    if has_fix and tool in _VERIFY_TOOLS:
        return "verify"
    if tool in _TRIAGE_TOOLS and not history:
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
    """Reward correct SRE workflow ordering, penalize skipping or going backward."""
    current_order = PHASE_ORDER.get(current_phase, 1)
    past_phases = [detect_phase(h.get("tool", ""), history[:i]) for i, h in enumerate(history)]

    if not past_phases:
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
        phase_score() on any API error so training stays stable.
        """
        deterministic = phase_score(detect_phase(tool, history), history)

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
  6. document — diagnose_root_cause → done

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
            # Return 100% LLM reward, no deterministic fallback
            print(f"  [Judge] {tool}: LLM={llm_reward:+.2f} | {rationale}")
            return round(llm_reward, 3)
        except Exception as e:
            # Rate limits / network errors → fall back silently to 0.0 (no deterministic)
            if "429" not in str(e) and "503" not in str(e):
                print(f"  [Judge] Step reward LLM error: {e}")
            return 0.0

    def evaluate_terminal(
        self, scenario: dict, history: list, stack_healthy: bool, sla_status: dict
    ) -> tuple[float, str]:
        fallback_score, fallback_feedback = self._fallback_evaluate(
            scenario, history, stack_healthy, sla_status
        )

        if not self.client:
            return fallback_score, fallback_feedback

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

Evaluate the agent's incident response based on:
1. Did they use proper SRE workflow (Triage → Investigate → Diagnose → Fix → Verify)?
2. Did they successfully diagnose and mitigate the issue?
3. Did they avoid reckless or destructive actions?
4. Did they use the expected fix tools if needed?

Provide a score (0.0 = terrible/destructive, 1.0 = perfect) and brief feedback."""

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
            score = float(data.get("score", fallback_score))
            feedback = str(data.get("feedback", fallback_feedback))
            return max(0.0, min(1.0, score)), feedback
        except Exception as e:
            print(f"Gemini judge failed: {e}")
            return fallback_score, fallback_feedback

    def _fallback_evaluate(
        self, scenario: dict, history: list, stack_healthy: bool, sla_status: dict
    ) -> tuple[float, str]:
        """Programmatic evaluation with expected_fix checking."""
        tools_used = [h["tool"] for h in history]

        score = TERMINAL_BASE_SCORE
        feedback = "Completed."

        # 1. Did they actually fix it?
        if stack_healthy:
            score += TERMINAL_HEALTHY_BONUS
            feedback = "Successfully restored cluster health."
        else:
            score += TERMINAL_UNHEALTHY_PENALTY
            feedback = "Cluster remained unhealthy."

        # 2. Did they use the expected fix tools?
        expected_fixes = scenario.get("expected_fix", [])
        if expected_fixes:
            used_correct = [f for f in expected_fixes if f in tools_used]
            if used_correct:
                score += TERMINAL_EXPECTED_FIX_BONUS
                feedback += f" Correctly used {', '.join(used_correct)}."
            else:
                score += TERMINAL_WRONG_FIX_PENALTY
                feedback += f" Did not use expected fix ({', '.join(expected_fixes)})."

        # 3. Did they do reckless things?
        if "redis_flush_db" in tools_used and scenario.get("layer") != "cache":
            score += TERMINAL_UNNECESSARY_FLUSH
            feedback += " Unnecessarily flushed Redis."
        if "docker_restart" in tools_used:
            if len(history) < TERMINAL_RECKLESS_THRESHOLD:
                score += TERMINAL_RECKLESS_RESTART
                feedback += " Reckless restart without investigation."

        # 4. Did they find the root cause?
        for h in history:
            if h["tool"] == "diagnose_root_cause":
                score += TERMINAL_ROOT_CAUSE_BONUS
                feedback += " Good logging of root cause."
                break  # only count once

        # 5. SLA Penalty
        if sla_status.get("sla_status") == "VIOLATED":
            score += TERMINAL_SLA_VIOLATED
            feedback += " SLA violated due to slow response."

        return max(0.0, min(1.0, score)), feedback
