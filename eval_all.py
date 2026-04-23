import time
from server.llm_designer import WARMUP_SCENARIOS, MEDIUM_SCENARIOS, HARD_SCENARIOS
from server.llm_judge import LLMJudge

all_scenarios = WARMUP_SCENARIOS + MEDIUM_SCENARIOS + HARD_SCENARIOS
judge = LLMJudge()

print("============================================================")
print("Running Gemini LLM Judge Terminal Evaluation for All Scenarios")
print("============================================================")
print("Note: Simulating a 'successful' agent trajectory (with appropriate")
print("diagnosis and expected fixes) for each scenario since local test")
print("Docker containers are currently down.\n")

for sc in all_scenarios:
    sc_dump = sc.model_dump()
    expected_fixes = sc_dump.get('expected_fix', [])
    
    # Create a simulated perfect trajectory
    if sc_dump['layer'] == 'database':
        inv_tool = 'pg_stat_activity'
    elif sc_dump['layer'] == 'cache':
        inv_tool = 'redis_info'
    else:
        inv_tool = 'docker_logs'
        
    history = [
        {"tool": inv_tool, "args": {}, "output": "Found the anomaly matching the alert.", "reward": 0.2},
    ]
    for fix in expected_fixes:
        history.append({"tool": fix, "args": {}, "output": f"Successfully applied {fix}.", "reward": 0.2})
        
    history.append({"tool": "curl_endpoint", "args": {"url": "http://localhost:5001/health"}, "output": "HTTP 200 OK", "reward": 0.1})
    history.append({"tool": "diagnose_root_cause", "args": {}, "output": "Root cause documented and resolved.", "reward": 0.1})
    
    sla_status = {"sla_status": "OK", "revenue_loss_usd": 100, "downtime_minutes": 1.0}
    
    # We may hit 429 rate limit if we do all 12 too fast, so add simple retry
    attempts = 3
    while attempts > 0:
        score, feedback = judge.evaluate_terminal(sc_dump, history, stack_healthy=True, sla_status=sla_status)
        if "google.api_core.exceptions" in feedback or "Completed." in feedback:
           # means it fell back or failed
           time.sleep(30)
           attempts -= 1
        else:
           break

    print(f"[{sc_dump['layer'].upper()}] {sc_dump['name']}")
    print(f"  Score:    {score}/1.0")
    print(f"  Feedback: {feedback}\n")
    time.sleep(5)  # Pause to respect 5 RPM free-tier quota spread out
