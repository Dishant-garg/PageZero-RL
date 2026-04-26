from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field, ConfigDict
from typing import Literal, Dict, Any, Optional, List

ToolName = Literal[
    # Layer 1: Monitoring & Triage
    "check_alerts", 
    "get_service_metrics", 
    "get_error_rate",
    
    # Layer 2: Application
    "read_app_logs", 
    "search_logs", 
    "get_recent_deploys", 
    "rollback_deploy", 
    "curl_endpoint",
    
    # Layer 3: PostgreSQL
    "pg_stat_activity", 
    "pg_locks", 
    "pg_explain_analyze", 
    "pg_stat_statements",
    "pg_cancel_query", 
    "pg_create_index", 
    "pg_vacuum", 
    "pg_show_tables",
    
    # Layer 4: Redis
    "redis_info", 
    "redis_slowlog", 
    "redis_keys", 
    "redis_flush_db", 
    "redis_get_key",
    
    # Layer 5: Infrastructure
    "docker_ps", 
    "docker_stats", 
    "docker_restart", 
    "docker_logs", 
    "check_disk_usage",
    
    # Resolution
    "diagnose_root_cause", 
    "write_postmortem",
    "done"
]

class PageZeroAction(Action):
    """The tools a Staff SRE uses to debug across the full stack."""
    tool: ToolName = Field(..., description="Action tool to perform")
    args: Dict[str, Any] = Field(default_factory=dict, description="Arguments for the tool")


class PageZeroObservation(Observation):
    """The environment state and telemetry data returned after each action."""
    # Forward/backward compatibility guard:
    # tolerate unknown observation keys when backend/client versions differ.
    model_config = ConfigDict(extra="allow")
    tool_output: str = Field(default="", description="Result of the last tool call")
    active_alerts: List[str] = Field(default_factory=list, description="Current firing PagerDuty alerts")
    sla_status: str = Field(default="OK", description="Current SLA status (OK/DEGRADED/VIOLATED)")
    revenue_loss_usd: float = Field(default=0.0, description="Cumulative revenue lost due to downtime")
    downtime_minutes: float = Field(default=0.0, description="Duration of the incident")
    step: int = Field(default=0, description="Current step number")
    max_steps: int = Field(default=15, description="Maximum allowed steps before failure")
    hint: Optional[str] = Field(default=None, description="Optional feedback or error hints")
    phase_history: List[str] = Field(default_factory=list, description="SRE phases executed so far")
    # Termination flag — true when the episode ended (whether successful or not).
    # Kept for backward compatibility; do NOT use this as a "did we fix it?"
    # signal — the wrapper used to do that and silently mislabeled every
    # premature `done` as resolved. Read ``stack_healthy`` instead.
    is_done: bool = Field(default=False, description="Whether the episode terminated (legacy; use stack_healthy for resolution)")
    # Real, programmatic resolution signal: True only when the backend health
    # poll AND (when available) the LLM judge BOTH confirm the stack is fixed.
    # Populated on the terminal step; auto-included after any fix-shaped tool.
    stack_healthy: Optional[bool] = Field(default=None, description="Programmatic + judge-confirmed stack health (None if not yet evaluated)")
    final_score: Optional[float] = Field(default=None, description="Terminal reward score")
    reward: float = Field(default=0.0, description="Step reward")
    # Last-turn judge feedback / hint, surfaced so the next user prompt can
    # show the agent why the previous step earned its reward.
    judge_feedback: Optional[str] = Field(default=None, description="One-line judge feedback for the latest step")
    # SRE workflow phase the most recent action was classified into.
    phase: Optional[str] = Field(default=None, description="Detected SRE phase of the latest action")
    # True if the env auto-classified this step as a fix attempt (mutation tool).
    is_fix_step: bool = Field(default=False, description="Whether the env treated the last action as a fix attempt")
    # Repeat counter for the most recent command (1 = first call, 2 = second, etc.)
    repeat_count: int = Field(default=1, description="How many times the latest tool/args pair has been called this episode")
    done: bool = Field(default=False, description="Is the incident investigation complete?")

class PageZeroState(State):
    """Internal state of the PageZero Environment."""
    episode_id: str = Field(default="", description="ID of the current episode")
    step_count: int = Field(default=0, description="Current step number")
    difficulty: float = Field(default=0.15, description="Current curriculum difficulty")
    scenario_name: str = Field(default="None", description="Name of current scenario")
    is_resolved: bool = Field(default=False, description="Whether the incident was successfully resolved")
    cumulative_reward: float = Field(default=0.0, description="Cumulative reward for the episode")

