"""PageZero SRE Inference — Stateless RL Agent (workspace_agent pattern).

Each step sends a SINGLE fresh prompt. No multi-turn chat history.
The model sees: Alert → SLA → Latest Output → Reward → History Summary → Recovery Hint.

Output format:
  [START] task=<scenario> env=PageZero model=<model>
  [STEP]  step=<n> action=<tool(args)> reward=<r> done=<bool> error=<e|null>
  [END]   success=<bool> steps=<n> score=<s> rewards=<r1,r2,...>

Environment variables:
  GEMINI_API_KEY          — required
  PZ_MODEL                — model name (default: gemini-2.5-flash)
  PZ_TEMPERATURE          — generation temp (default: 0.3)
  PZ_MAX_API_RETRIES      — API retry count (default: 3)
  PZ_MAX_SAME_ACTION      — streak before loop-break (default: 2)
  PZ_SUCCESS_THRESHOLD    — score threshold for success (default: 0.6)
  PZ_EPISODES             — number of episodes to run (default: 1)
"""

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

# ── Project root on path ───────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from google import genai
from google.genai import types

from server.PageZero_environment import PageZeroEnvironment
from models import PageZeroAction

# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════
MODEL_NAME = os.getenv("PZ_MODEL", "gemini-2.5-flash")
TEMPERATURE = float(os.getenv("PZ_TEMPERATURE", "0.3"))
MAX_API_RETRIES = int(os.getenv("PZ_MAX_API_RETRIES", "3"))
MAX_SAME_ACTION_STREAK = int(os.getenv("PZ_MAX_SAME_ACTION", "2"))
RECENT_ACTION_WINDOW = 6
SUCCESS_THRESHOLD = float(os.getenv("PZ_SUCCESS_THRESHOLD", "0.6"))

# ═══════════════════════════════════════════════════════════════════
# Tool Specification (documents required args for validation)
# ═══════════════════════════════════════════════════════════════════
TOOLS_SPEC = {
    # Triage
    "check_alerts":       {"required": []},
    "get_service_metrics":{"required": []},    # optional: service
    "get_error_rate":     {"required": []},
    # Application
    "read_app_logs":      {"required": []},    # optional: service, lines
    "search_logs":        {"required": ["pattern"]},
    "get_recent_deploys": {"required": []},
    "rollback_deploy":    {"required": []},
    "curl_endpoint":      {"required": ["url"]},
    # PostgreSQL
    "pg_stat_activity":   {"required": []},
    "pg_locks":           {"required": []},
    "pg_explain_analyze": {"required": ["query"]},
    "pg_stat_statements": {"required": []},
    "pg_cancel_query":    {"required": ["pid"]},
    "pg_create_index":    {"required": ["table", "column"]},
    "pg_vacuum":          {"required": ["table"]},
    "pg_show_tables":     {"required": []},
    # Redis
    "redis_info":         {"required": []},
    "redis_slowlog":      {"required": []},
    "redis_keys":         {"required": []},    # optional: pattern
    "redis_flush_db":     {"required": []},
    "redis_get_key":      {"required": ["key"]},
    # Infrastructure
    "docker_ps":          {"required": []},
    "docker_stats":       {"required": ["container"]},
    "docker_restart":     {"required": ["container"]},
    "docker_logs":        {"required": ["container"]},
    "check_disk_usage":   {"required": []},
    # Resolution
    "diagnose_root_cause":{"required": ["root_cause"]},
    "done":               {"required": []},
}

# ═══════════════════════════════════════════════════════════════════
# System Prompt (Stateless — sent on every call)
# ═══════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """Return ONLY a valid JSON object. No text, no markdown.

You are a Principal SRE autonomous agent responding to production incidents.
Architecture: PostgreSQL (pagezero-postgres-1), Redis (pagezero-redis-1), Flask app (pagezero-app-1).

SRE Workflow (follow in order):
  1. TRIAGE:      check_alerts, get_error_rate, get_service_metrics
  2. INVESTIGATE: read_app_logs, docker_ps, docker_logs, pg_stat_activity, redis_info, redis_keys, search_logs, check_disk_usage
  3. DIAGNOSE:    pg_explain_analyze, pg_stat_statements, get_recent_deploys
  4. FIX:         pg_cancel_query, pg_create_index, pg_vacuum, redis_flush_db, docker_restart, rollback_deploy
  5. VERIFY:      curl_endpoint, pg_stat_activity, redis_info, get_service_metrics
  6. DOCUMENT:    diagnose_root_cause → done

TOOL ARGUMENT RULES (follow exactly or you will get errors):
  - curl_endpoint:        REQUIRES {"url": "http://localhost:5001/health"}
  - docker_logs:          REQUIRES {"container": "pagezero-app-1"} (or postgres-1, redis-1)
  - docker_restart:       REQUIRES {"container": "pagezero-app-1"}
  - docker_stats:         REQUIRES {"container": "pagezero-app-1"}
  - pg_cancel_query:      REQUIRES {"pid": <number>} from pg_stat_activity
  - pg_create_index:      REQUIRES {"table": "orders", "column": "user_email"}
  - pg_vacuum:            REQUIRES {"table": "orders"}
  - pg_explain_analyze:   REQUIRES {"query": "SELECT ..."}
  - redis_get_key:        REQUIRES {"key": "some_key"}
  - search_logs:          REQUIRES {"pattern": "error_text"}
  - diagnose_root_cause:  REQUIRES {"root_cause": "one-sentence summary"}

CRITICAL RULES:
  - If a tool returns "ERROR: missing ...", you MUST fix the args or switch tools.
  - NEVER repeat the exact same tool+args if it returned an error.
  - NEVER call redis_flush_db unless Redis is the confirmed root cause.
  - NEVER call docker_restart before step 3 — investigate first.
  - get_service_metrics optionally takes {"service": "app"} or {"service": "redis"}.
  - check_alerts, get_error_rate, docker_ps, redis_info need NO args.

ANTI-PATTERNS THAT CAUSE ERRORS:
  - ❌ Calling docker_logs without {"container": "..."} — will ERROR
  - ❌ Calling pg_cancel_query without {"pid": ...} — will ERROR
  - ❌ Repeating same failed tool 3+ times — indicates wrong approach
  - ❌ Calling redis_flush_db as first action — reckless, investigate first

GOOD PATTERNS:
  - ✅ check_alerts → redis_info → redis_keys → redis_flush_db → redis_info → diagnose_root_cause → done
  - ✅ check_alerts → pg_stat_activity → pg_locks → pg_cancel_query → curl_endpoint → done
  - ✅ Always read error output carefully and fix arguments before retrying

Output format: {"tool": "<name>", "args": {...}}
"""


# ═══════════════════════════════════════════════════════════════════
# Logging (workspace_agent format)
# ═══════════════════════════════════════════════════════════════════
def _flat(text: Optional[str]) -> str:
    if text is None:
        return ""
    return str(text).replace("\n", "\\n").replace("\r", "\\r")


def _compact(text: Optional[str], max_chars: int = 600) -> str:
    if not text:
        return ""
    s = str(text).strip()
    return s[:max_chars - 3] + "..." if len(s) > max_chars else s


def log_start(task: str, env_name: str, model: str) -> None:
    print(f"[START] task={_flat(task)} env={_flat(env_name)} model={_flat(model)}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool,
             error: Optional[str] = None) -> None:
    done_s = "true" if done else "false"
    err_s = "null" if not error else _flat(error)
    print(f"[STEP] step={step} action={_flat(action)} reward={reward:.2f} "
          f"done={done_s} error={err_s}", flush=True)


def log_end(success: bool, steps: int, score: float,
            rewards: List[float]) -> None:
    s = "true" if success else "false"
    r = ",".join(f"{v:.2f}" for v in rewards)
    print(f"[END] success={s} steps={steps} score={score:.2f} rewards={r}",
          flush=True)


# ═══════════════════════════════════════════════════════════════════
# Loop-Breaking & Recovery
# ═══════════════════════════════════════════════════════════════════
def _action_signature(tool: str, args: Dict[str, Any]) -> str:
    return f"{tool}:{json.dumps(args, sort_keys=True, separators=(',', ':'))}"


def _choose_recovery_action(
    last_obs: str,
    recent_actions: List[Dict[str, Any]],
    scenario_layer: str,
) -> Tuple[str, Dict[str, Any], str]:
    """Pick a sensible recovery action based on context."""
    obs_lower = (last_obs or "").lower()

    # Specific error recovery
    if "missing 'url'" in obs_lower:
        return ("curl_endpoint",
                {"url": "http://localhost:5001/health"},
                "recovery:fix_missing_url")
    if "missing 'container'" in obs_lower:
        return "docker_ps", {}, "recovery:list_containers"
    if "missing 'pid'" in obs_lower:
        return "pg_stat_activity", {}, "recovery:get_pids"
    if "missing 'table'" in obs_lower or "missing 'column'" in obs_lower:
        return "pg_show_tables", {}, "recovery:list_tables"
    if "missing 'root_cause'" in obs_lower or "missing 'diagnosis'" in obs_lower:
        return ("diagnose_root_cause",
                {"root_cause": "Automated diagnosis — root cause under investigation."},
                "recovery:fix_diagnose_args")

    # Layer-based fallback
    if scenario_layer == "cache":
        return "redis_info", {}, "recovery:check_redis"
    if scenario_layer == "database":
        return "pg_stat_activity", {}, "recovery:check_db"

    # Last resort — explore from the top
    recent_tools = [a.get("tool") for a in (recent_actions or [])]
    if "check_alerts" not in recent_tools:
        return "check_alerts", {}, "recovery:start_triage"

    return "docker_ps", {}, "recovery:default"


def _apply_loop_breaker(
    tool: str,
    args: Dict[str, Any],
    recent_actions: List[Dict[str, Any]],
    last_obs: str,
    scenario_layer: str,
) -> Tuple[str, Dict[str, Any], Optional[str]]:
    """Intercept repeated actions to break out of stuck loops."""
    if not recent_actions:
        return tool, args, None

    proposed_sig = _action_signature(tool, args)

    # Count consecutive identical actions
    streak = 0
    for a in reversed(recent_actions):
        if a.get("signature") == proposed_sig:
            streak += 1
        else:
            break

    # Break if same exact action repeated too many times
    if streak >= MAX_SAME_ACTION_STREAK:
        r_tool, r_args, reason = _choose_recovery_action(
            last_obs, recent_actions, scenario_layer
        )
        return r_tool, r_args, f"loop_break:{reason}"

    # Break if the last action returned an error and this is the same tool
    if (recent_actions
            and recent_actions[-1].get("had_error")
            and recent_actions[-1].get("tool") == tool):
        r_tool, r_args, reason = _choose_recovery_action(
            last_obs, recent_actions, scenario_layer
        )
        return r_tool, r_args, f"error_recovery:{reason}"

    return tool, args, None


# ═══════════════════════════════════════════════════════════════════
# Stateless Prompt Builder (workspace_agent pattern)
# ═══════════════════════════════════════════════════════════════════
_ACTION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "tool": {"type": "STRING",
                 "description": "Tool name from the SRE toolkit."},
        "args": {"type": "OBJECT",
                 "description": "Arguments dict for the tool."},
    },
    "required": ["tool"],
}


def _build_recovery_hint(
    latest_output: str,
    recent_actions: List[Dict[str, Any]],
) -> str:
    """Generate a dynamic recovery hint based on the last output."""
    lowered = (latest_output or "").lower()

    if "missing" in lowered and "container" in lowered:
        return ("docker_logs/docker_stats/docker_restart REQUIRE "
                '{"container": "pagezero-app-1"}. Use docker_ps first to list containers.')
    if "missing" in lowered and "pid" in lowered:
        return "pg_cancel_query REQUIRES {\"pid\": <number>}. Use pg_stat_activity first to find PIDs."
    if "missing" in lowered and ("table" in lowered or "column" in lowered):
        return "This tool requires table/column args. Use pg_show_tables first."
    if "missing" in lowered and "url" in lowered:
        return 'curl_endpoint REQUIRES {"url": "http://localhost:5001/health"}.'
    if "error" in lowered:
        return "Tool failed. Read the error carefully, fix arguments, or switch to a different tool."

    # Check for exploration stall
    if len(recent_actions) >= 4:
        recent_tools = {a.get("tool") for a in recent_actions[-4:]}
        if recent_tools <= {"check_alerts", "get_error_rate", "get_service_metrics", "docker_ps"}:
            return "You have been in TRIAGE too long. Move to INVESTIGATE phase: read_app_logs, redis_info, pg_stat_activity."

    if len(recent_actions) >= 6:
        error_count = sum(1 for a in recent_actions[-4:] if a.get("had_error"))
        if error_count >= 2:
            return "Multiple recent errors. Switch to a completely different tool class (explore vs fix vs verify)."

    return "Follow the SRE workflow: Triage → Investigate → Diagnose → Fix → Verify → Document."


def _build_user_prompt(
    step: int,
    max_steps: int,
    alert: str,
    sla_status: str,
    revenue_loss: float,
    downtime_min: float,
    latest_output: str,
    last_reward: float,
    history: List[str],
    phase_history: List[str],
    recovery_hint: str,
) -> str:
    """Build a fresh, stateless prompt for every step (no chat history)."""
    budget = max_steps - step
    history_block = "\n".join(history[-5:]) if history else "No prior steps."
    output_block = _compact(latest_output) or "No observation yet."

    # Show which tools have been used so far (like workspace_agent's file listing)
    tools_used = sorted(set(phase_history)) if phase_history else []
    tools_block = ", ".join(tools_used) if tools_used else "None yet"

    prompt = f"""Step: {step}/{max_steps} ({budget} remaining)

🚨 Active Alert: {alert}

SLA Status: {sla_status} | Revenue Loss: ${revenue_loss:,.0f} | Downtime: {downtime_min:.1f}min

Latest Tool Output:
{output_block}

Last Reward: {last_reward:+.2f}

Tools Used So Far: {tools_block}

History:
{history_block}

Recovery Hint: {recovery_hint}

Choose your next action."""
    return prompt


def _choose_action(
    client: genai.Client,
    user_prompt: str,
) -> Tuple[str, Dict[str, Any]]:
    """Call Gemini with a SINGLE stateless prompt (no conversation history)."""
    last_err: Optional[Exception] = None
    for attempt in range(MAX_API_RETRIES):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[
                    {"role": "user", "parts": [{"text": user_prompt}]}
                ],
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=_ACTION_SCHEMA,
                    temperature=TEMPERATURE,
                ),
            )
            data = json.loads(response.text)
            tool = str(data.get("tool", "")).strip()
            args = data.get("args") or {}
            if not isinstance(args, dict):
                args = {}
            if tool not in TOOLS_SPEC:
                raise ValueError(f"Unknown tool: {tool!r}")
            return tool, args

        except Exception as e:
            last_err = e
            err_str = str(e)
            if "429" in err_str and attempt < MAX_API_RETRIES - 1:
                time.sleep(35)
            elif "503" in err_str and attempt < MAX_API_RETRIES - 1:
                time.sleep(5)
            elif attempt < MAX_API_RETRIES - 1:
                time.sleep(2)
            else:
                break

    raise RuntimeError(
        f"Action selection failed after {MAX_API_RETRIES} retries: {last_err}"
    )


# ═══════════════════════════════════════════════════════════════════
# Episode Runner
# ═══════════════════════════════════════════════════════════════════
def run_episode(client: genai.Client, env: PageZeroEnvironment) -> Dict[str, Any]:
    """Run a single SRE incident episode. Returns summary dict."""
    rewards: List[float] = []
    steps_taken = 0
    final_score = 0.0
    success = False
    history_summary: List[str] = []
    recent_actions: List[Dict[str, Any]] = []
    last_reward = 0.0

    # ── Reset ──
    obs = env.reset()
    scenario = getattr(env, "_scenario", {}) or {}
    scenario_name = scenario.get("name", "unknown")
    scenario_layer = scenario.get("layer", "cross_layer")
    alert_text = scenario.get("alert", "Unknown production anomaly")

    log_start(task=scenario_name, env_name="PageZero", model=MODEL_NAME)

    max_steps = getattr(obs, "max_steps", 15)

    for step in range(1, max_steps + 1):
        if obs.is_done:
            break

        # ── Extract structured state from observation ──
        latest_output = obs.tool_output or ""
        sla_status = obs.sla_status or "OK"
        revenue_loss = obs.revenue_loss_usd or 0.0
        downtime_min = obs.downtime_minutes or 0.0
        phase_history = obs.phase_history or []

        try:
            # 1. Build recovery hint
            recovery_hint = _build_recovery_hint(latest_output, recent_actions)

            # 2. Build fresh stateless prompt
            user_prompt = _build_user_prompt(
                step=step,
                max_steps=max_steps,
                alert=alert_text,
                sla_status=sla_status,
                revenue_loss=revenue_loss,
                downtime_min=downtime_min,
                latest_output=latest_output,
                last_reward=last_reward,
                history=history_summary,
                phase_history=phase_history,
                recovery_hint=recovery_hint,
            )

            # 3. Call model (stateless — no conversation history)
            proposed_tool, proposed_args = _choose_action(client, user_prompt)

            # 4. Loop-breaking
            tool, args, override_reason = _apply_loop_breaker(
                proposed_tool,
                proposed_args,
                recent_actions,
                latest_output,
                scenario_layer,
            )

            # 5. Step the environment
            action_obj = PageZeroAction(tool=tool, args=args)
            step_result = env.step(action_obj)
            obs = step_result.observation
            reward = float(step_result.reward or 0.0)
            done = bool(step_result.done)

            # 6. Detect errors in output
            error_str = None
            tool_output = obs.tool_output or ""
            if tool_output.startswith("ERROR"):
                error_str = _compact(tool_output, 200)

            # 7. Format action string for logging
            action_str = f"{tool}({json.dumps(args, separators=(',', ':'))})"
            if override_reason:
                action_str = f"[{override_reason}] {action_str}"

            # 8. Record everything
            rewards.append(reward)
            last_reward = reward
            steps_taken = step

            recent_actions.append({
                "tool": tool,
                "signature": _action_signature(tool, args),
                "had_error": bool(error_str),
                "reward": reward,
            })
            if len(recent_actions) > RECENT_ACTION_WINDOW:
                recent_actions.pop(0)

            history_summary.append(
                f"Step {step}: {tool}({json.dumps(args, separators=(',', ':'))}) "
                f"→ reward={reward:+.2f}"
                + (f" [ERROR: {_compact(tool_output, 80)}]" if error_str else "")
            )

            log_step(step, action_str, reward, done, error_str)

            if done:
                break

        except Exception as exc:
            action_str = f"error({_compact(str(exc), 120)})"
            log_step(step, action_str, 0.0, True, _compact(str(exc), 200))
            steps_taken = step
            break

    # ── Terminal scoring (strictly inside (0,1) for OpenEnv validator) ──
    final_score = obs.final_score if obs.final_score is not None else 0.0
    final_score = max(0.01, min(0.99, final_score))
    success = final_score >= SUCCESS_THRESHOLD

    log_end(success, steps_taken, final_score, rewards)

    return {
        "task": scenario_name,
        "success": success,
        "steps": steps_taken,
        "score": final_score,
        "rewards": rewards,
    }


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main() -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or "your_" in api_key:
        print("Error: Set GEMINI_API_KEY in .env")
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    env = PageZeroEnvironment()

    episodes = int(os.getenv("PZ_EPISODES", "1"))
    results = []

    for ep in range(1, episodes + 1):
        print(f"\n{'='*60}")
        print(f"  Episode {ep}/{episodes}")
        print(f"{'='*60}")
        result = run_episode(client, env)
        results.append(result)

    # ── Summary ──
    if len(results) > 1:
        avg_score = sum(r["score"] for r in results) / len(results)
        wins = sum(1 for r in results if r["success"])
        print(f"\n{'='*60}")
        print(f"  SUMMARY: {wins}/{len(results)} successful  "
              f"avg_score={avg_score:.2f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
