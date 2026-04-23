import os
import json
import time
from dotenv import load_dotenv

load_dotenv()

# Add project root to path
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from google import genai
from google.genai import types

from server.PageZero_environment import PageZeroEnvironment
from models import PageZeroAction

def play_environment():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or "your_" in api_key:
        print("Please set GEMINI_API_KEY in .env")
        return

    client = genai.Client(api_key=api_key)
    env = PageZeroEnvironment()
    
    print("="*60)
    print("Initializing PageZero Environment...")
    print("="*60)
    
    obs = env.reset()
    
    # Simple structured output schema matching our Action
    schema = {
        "type": "OBJECT",
        "properties": {
            "tool": {
                "type": "STRING",
                "description": "Name of the tool to run. MUST be one of: check_alerts, get_service_metrics, get_error_rate, read_app_logs, search_logs, get_recent_deploys, rollback_deploy, curl_endpoint, pg_stat_activity, pg_locks, pg_explain_analyze, pg_stat_statements, pg_cancel_query, pg_create_index, pg_vacuum, pg_show_tables, redis_info, redis_slowlog, redis_keys, redis_flush_db, redis_get_key, docker_ps, docker_stats, docker_restart, docker_logs, check_disk_usage, diagnose_root_cause, done."
            },
            "args": {
                "type": "OBJECT",
                "description": "Arguments for the tool (e.g. {'query': 'SELECT...'} or {'container': 'pagezero-app-1'})."
            }
        },
        "required": ["tool"]
    }
    
    system_prompt = """You are a Principal SRE autonomous agent responding to a production incident.
You have full access to PostgreSQL, Redis, and a Flask app running in Docker containers.

Follow this SRE workflow strictly:
  1. TRIAGE: Use check_alerts, get_error_rate, or get_service_metrics to understand the blast radius.
  2. INVESTIGATE: Use read_app_logs, docker_ps, docker_logs, pg_stat_activity, redis_info, etc.
  3. DIAGNOSE: Use pg_explain_analyze, pg_stat_statements, search_logs to pinpoint root cause.
  4. FIX: Apply the targeted fix (pg_cancel_query, redis_flush_db, docker_restart, pg_vacuum, etc.).
  5. VERIFY: Confirm the fix worked with curl_endpoint, pg_stat_activity, redis_info.
  6. DOCUMENT: Call diagnose_root_cause with a short summary, then call done.

CRITICAL RULES:
- Always pass required args. docker_logs, docker_restart, docker_stats require {"container": "pagezero-app-1"} (or postgres-1/redis-1).
- pg_cancel_query requires {"pid": <pid_number>} from pg_stat_activity output.
- pg_create_index requires {"table": "orders", "column": "user_email"}.
- pg_vacuum requires {"table": "orders"}.
- curl_endpoint requires {"url": "http://localhost:5001/health"}.
- Do NOT repeat the same tool+args twice in a row — it wastes steps.
- Never call redis_flush_db unless Redis is the confirmed root cause.
- Never call docker_restart in step 1 or 2 — investigate first.

Return your action as JSON matching the schema provided."""
    
    history = []
    
    while True:
        # Prepare observation for the model
        obs_data = obs.model_dump()
        obs_str = json.dumps(obs_data, indent=2)
        print(f"\n[ENVIRONMENT OBSERVED]\n{obs_str}\n")
        
        if obs.is_done:
            print("="*60)
            print(f"INCIDENT CLOSED!")
            print(f"Final Score: {obs.final_score}")
            print(f"Feedback: {obs.hint}")
            print("="*60)
            break
            
        # Build user message — include scenario hint on step 0 as a diagnostic nudge
        scenario_hint = getattr(env, '_scenario', {}) or {}
        hint_str = scenario_hint.get("hint", "")
        
        user_msg = f"Observation:\n{obs_str}\n"
        if hint_str and len(history) == 0:
            user_msg += f"\n💡 Diagnostic Hint: {hint_str}\n"
        user_msg += "\nWhat is your next action?"
        history.append({"role": "user", "parts": [{"text": user_msg}]})
        
        try:
            print("Thinking...")
            # Use JSON mode for reliable output with retry logic
            response = None
            for attempt in range(3):
                try:
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=history,
                        config=types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            response_mime_type="application/json",
                            response_schema=schema,
                            temperature=0.2
                        )
                    )
                    break
                except Exception as api_err:
                    err_str = str(api_err)
                    if "503" in err_str and attempt < 2:
                        print(f"API overloaded, retrying in 5 seconds... ({attempt+1}/3)")
                        time.sleep(5)
                    elif "429" in err_str and attempt < 2:
                        print(f"Rate limit hit, retrying in 35 seconds... ({attempt+1}/3)")
                        time.sleep(35)
                    else:
                        raise api_err
            
            if not response:
                 raise Exception("Failed to get response after retries.")
                 
            action_data = json.loads(response.text)
            action = PageZeroAction(**action_data)
            
            history.append({"role": "model", "parts": [{"text": response.text}]})
            
            print(f"> Taking action: {action.tool} with args {action.args}")
            
            # Step the environment
            step_res = env.step(action)
            obs = step_res.observation
            print(f"  (Reward: {step_res.reward:.2f})")
            
        except Exception as e:
            print(f"\n[ERROR] Failed during LLM call or step: {e}")
            break

if __name__ == "__main__":
    play_environment()
