_TRIAGE_TOOLS = {"check_alerts", "get_service_metrics", "get_error_rate"}
_INVESTIGATE_TOOLS = {"pg_stat_activity", "pg_locks", "redis_info", "redis_slowlog",
                       "read_app_logs", "search_logs", "docker_ps", "docker_stats",
                       "pg_show_tables", "check_disk_usage", "redis_keys", "redis_get_key",
                       "docker_logs", "curl_endpoint"}
_DIAGNOSE_TOOLS = {"pg_explain_analyze", "pg_stat_statements", "get_recent_deploys"}
_FIX_TOOLS = {"pg_cancel_query", "pg_create_index", "pg_vacuum", "redis_flush_db",
              "docker_restart", "rollback_deploy"}
_VERIFY_TOOLS = {"get_service_metrics", "curl_endpoint", "pg_stat_activity", "redis_info"}
_DOCUMENT_TOOLS = {"diagnose_root_cause"}

PHASE_ORDER = {"triage": 0, "investigate": 1, "diagnose": 2, "fix": 3, "verify": 4, "document": 5}

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
    """Reward correct SRE workflow ordering, penalize skipping phases."""
    current_order = PHASE_ORDER.get(current_phase, 1)
    past_phases = [detect_phase(h.get("tool", ""), history[:i]) for i, h in enumerate(history)]
    
    if not past_phases:
        return 0.15 if current_phase == "triage" else -0.1

    max_past = max([PHASE_ORDER.get(p, 0) for p in past_phases] + [0])
    if current_order <= max_past + 1:
        return 0.10  # Correct order bonus
    else:
        return -0.20  # Skipped phases penalty

import os
import json
from dotenv import load_dotenv

load_dotenv()

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

    def get_phase_reward(self, tool: str, history: list) -> float:
        phase = detect_phase(tool, history)
        return phase_score(phase, history)

    def evaluate_terminal(self, scenario: dict, history: list, stack_healthy: bool, sla_status: dict) -> tuple[float, str]:
        fallback_score, fallback_feedback = self._fallback_evaluate(scenario, history, stack_healthy, sla_status)
        
        if not self.client:
            return fallback_score, fallback_feedback

        # We'll trim the history output if it's too long
        trimmed_history = []
        for h in history:
            trimmed = dict(h)
            out = str(trimmed.get("output", ""))
            if len(out) > 500:
                trimmed["output"] = out[:500] + "... [truncated]"
            trimmed_history.append(trimmed)

        prompt = f"""You are a Principal SRE Judge evaluating a junior agent's incident response trajectory.

Incident Scenario: {json.dumps(scenario.get('name'))}
Root Cause: {json.dumps(scenario.get('root_cause'))}
Terminal Cluster Health (Did they fix it?): {stack_healthy}
SLA Status: {json.dumps(sla_status)}

Trajectory History:
{json.dumps(trimmed_history, indent=2)}

Evaluate the agent's performance. Based on order of operations (Triage -> Investigate -> Fix -> Verify) and whether they successfully diagnosed and mitigated the issue without doing destructive non-related actions (like randomly flushing databases without reason).
Return exactly a JSON object in this format (and ONLY this):
{{
    "score": float between 0.0 and 1.0 (0 meaning terrible/destructive, 1.0 meaning perfect),
    "feedback": "Concise 1-2 sentence feedback explaining the score"
}}
"""
        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash", 
                contents=prompt
            )
            raw = response.text.replace("```json", "").replace("```", "").strip()
            data = json.loads(raw)
            score = float(data.get("score", fallback_score))
            feedback = str(data.get("feedback", fallback_feedback))
            return max(0.0, min(1.0, score)), feedback
        except Exception as e:
            print(f"Gemini judge failed: {e}")
            return fallback_score, fallback_feedback

    def _fallback_evaluate(self, scenario: dict, history: list, stack_healthy: bool, sla_status: dict) -> tuple[float, str]:
        # Programmatic fast-evaluation (as sandbox for hackathon if no LLM key)
        # Check if the expected fix is in the history
        tools_used = [h["tool"] for h in history]
        
        # Penalties logic
        score = 0.5
        feedback = "Completed."
        
        # Did they actually fix it?
        if stack_healthy:
            score += 0.4
            feedback = "Successfully restored cluster health."
        else:
            score -= 0.4
            feedback = "Cluster remained unhealthy."
            
        # Did they do reckless things?
        if "redis_flush_db" in tools_used and scenario.get("layer") != "cache":
            score -= 0.2
            feedback += " Unnecessarily flushed Redis."
        if "docker_restart" in tools_used:
             # Restarting containers without investigation is bad
             if len(history) < 3:
                 score -= 0.2
                 feedback += " Reckless restart without investigation."
                 
        # Did they find the root cause?
        for h in history:
            if h["tool"] == "diagnose_root_cause":
                score += 0.1
                feedback += " Good logging of root cause."
                
        # SLA Penalty
        if sla_status.get("sla_status") == "VIOLATED":
            score -= 0.1
            feedback += " SLA violated due to slow response."
            
        return max(0.0, min(1.0, score)), feedback
