"""
PageZero Configuration — Single source of truth for all tunable constants.

Change values here instead of digging through multiple files.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ═══════════════════════════════════════════════════════════════════
# Docker Container Names
# ═══════════════════════════════════════════════════════════════════
POSTGRES_CONTAINER = "pagezero-postgres-1"
REDIS_CONTAINER = "pagezero-redis-1"
APP_CONTAINER = "pagezero-app-1"

# ═══════════════════════════════════════════════════════════════════
# Database Configuration (from .env)
# ═══════════════════════════════════════════════════════════════════
DB_HOST = os.getenv("DB_HOST", "postgres")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_USER = os.getenv("DB_USER", "sre")
DB_PASSWORD = os.getenv("DB_PASSWORD", "sre123")
DB_NAME = os.getenv("DB_NAME", "production")
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# ═══════════════════════════════════════════════════════════════════
# Redis Configuration (from .env)
# ═══════════════════════════════════════════════════════════════════
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"

# ═══════════════════════════════════════════════════════════════════
# Application Configuration (from .env)
# ═══════════════════════════════════════════════════════════════════
APP_INTERNAL_PORT = int(os.getenv("APP_INTERNAL_PORT", "5000"))
APP_EXTERNAL_HOST = os.getenv("APP_EXTERNAL_HOST", "localhost")
APP_EXTERNAL_PORT = int(os.getenv("APP_EXTERNAL_PORT", "5001"))
APP_HEALTH_URL = f"http://{APP_EXTERNAL_HOST}:{APP_EXTERNAL_PORT}/health"

# ═══════════════════════════════════════════════════════════════════
# SLA / Revenue Settings
# ═══════════════════════════════════════════════════════════════════
REVENUE_RATE_PER_MINUTE = float(os.getenv("PZ_REVENUE_RATE", "3900.0"))
SLA_THRESHOLD_MINUTES = float(os.getenv("PZ_SLA_THRESHOLD", "5.0"))

# ═══════════════════════════════════════════════════════════════════
# Episode Settings
# ═══════════════════════════════════════════════════════════════════
DEFAULT_MAX_STEPS = int(os.getenv("PZ_MAX_STEPS", "15"))
INJECTION_WAIT_SECONDS = float(os.getenv("PZ_INJECT_WAIT", "3.0"))

# ═══════════════════════════════════════════════════════════════════
# Curriculum / Difficulty Progression
# ═══════════════════════════════════════════════════════════════════
STARTING_DIFFICULTY = 0.15
DIFFICULTY_STEP = 0.1
MAX_DIFFICULTY = 1.0
SUCCESS_THRESHOLD = 0.6          # avg score needed to advance difficulty
MIN_EPISODES_TO_ADVANCE = 5     # how many episodes in a layer before advancing

# Warmup probability at each difficulty tier
WARMUP_PROB_LOW = 0.7            # difficulty < 0.4 → 70% warmup
WARMUP_PROB_MID = 0.3            # difficulty < 0.7 → 30% warmup
WARMUP_PROB_HIGH = 0.1           # difficulty ≥ 0.7 → 10% warmup
WARMUP_THRESHOLD_LOW = 0.4
WARMUP_THRESHOLD_HIGH = 0.7

# Schema drift thresholds
DRIFT_MIN_DIFFICULTY_DB = 0.5
DRIFT_MIN_DIFFICULTY_CACHE = 0.6
DRIFT_PROBABILITY = 0.5          # 50% chance when conditions met
DRIFT_TRIGGER_STEP_DB = 5
DRIFT_TRIGGER_STEP_CACHE = 6

# Scenario selection
HARD_SCENARIO_THRESHOLD = 0.6   # difficulty > this → hard scenario

# ═══════════════════════════════════════════════════════════════════
# Phase-Based Reward Values (kube-sre-gym aligned: +0.2 correct, -0.3 skipped)
# ═══════════════════════════════════════════════════════════════════
REWARD_TRIAGE_FIRST = 0.20       # Triage as first action
REWARD_WRONG_FIRST = -0.15       # Non-triage as first action
REWARD_CORRECT_ORDER = 0.20      # Correct SRE phase ordering
REWARD_SKIPPED_PHASE = -0.30     # Skipped a phase
REWARD_BACKWARD_PHASE = -0.15    # Went backward in SRE workflow

# ═══════════════════════════════════════════════════════════════════
# Repeat-command penalty (escalating + circuit breaker, kube-sre-gym style)
# ═══════════════════════════════════════════════════════════════════
REWARD_REPEAT_2X = -0.30         # 2nd identical (tool, args) call
REWARD_REPEAT_3X = -0.50         # 3rd+ identical call → circuit breaker, output replaced

# ═══════════════════════════════════════════════════════════════════
# Output-format / parse-failure penalty
# ═══════════════════════════════════════════════════════════════════
REWARD_NO_TOOL_CALL = -0.5       # Completion did not parse to any valid tool call

# Tier-1 curriculum guard: block documentation tools before real triage/investigate
REWARD_EARLY_DOC_BLOCK = -0.5    # diagnose_root_cause / write_postmortem on step < 3
MIN_STEPS_BEFORE_DONE = int(os.getenv("PZ_MIN_STEPS_BEFORE_DONE", "3"))
MIN_STEPS_BEFORE_RESOLVE = int(os.getenv("PZ_MIN_STEPS_BEFORE_RESOLVE", "5"))
REQUIRE_DOCS_BEFORE_SUCCESS = os.getenv("PZ_REQUIRE_DOCS_BEFORE_SUCCESS", "1") == "1"
REWARD_DONE_UNRESOLVED = float(os.getenv("PZ_DONE_UNRESOLVED_PENALTY", "-0.4"))
REWARD_DONE_BEFORE_DOCS = float(os.getenv("PZ_DONE_BEFORE_DOCS_PENALTY", "-0.3"))

# ═══════════════════════════════════════════════════════════════════
# Terminal reward (high-variance, kube-sre-gym style)
# ═══════════════════════════════════════════════════════════════════
TERMINAL_RESOLVED_BASE = 1.0     # base bonus when both gates confirm resolution
TERMINAL_RESOLVED_DIFF_WEIGHT = 2.0   # difficulty-scaled multiplier
TERMINAL_RESOLVED_EFFICIENCY_WEIGHT = 2.0  # extra (1 - steps/max_steps) bonus
TERMINAL_TIMEOUT_FAIL_TOTAL = -2.0  # net total for timeout / unresolved-done

# ═══════════════════════════════════════════════════════════════════
# Terminal Evaluation Reward Values
# ═══════════════════════════════════════════════════════════════════
TERMINAL_BASE_SCORE = 0.3
TERMINAL_HEALTHY_BONUS = 0.25
TERMINAL_UNHEALTHY_PENALTY = -0.4
TERMINAL_UNNECESSARY_FLUSH = -0.2
TERMINAL_RECKLESS_RESTART = -0.2
TERMINAL_ROOT_CAUSE_BONUS = 0.1
TERMINAL_SLA_VIOLATED = -0.1
TERMINAL_EXPECTED_FIX_BONUS = 0.1      # agent used the correct fix
TERMINAL_WRONG_FIX_PENALTY = -0.15     # agent "fixed" with wrong tool
TERMINAL_RECKLESS_THRESHOLD = 3        # min steps before restart is OK

# Efficiency Penalties (penalize wasteful, error-prone agents)
TERMINAL_ERROR_PENALTY_PER = -0.05     # per step that returned ERROR
TERMINAL_REPEAT_PENALTY_PER = -0.03    # per redundant repeat of same tool
TERMINAL_EFFICIENCY_WEIGHT = 0.15      # bonus scaled by (1 - steps_used/max_steps)

# ═══════════════════════════════════════════════════════════════════
# Train.py Penalties
# ═══════════════════════════════════════════════════════════════════
TRAIN_INVALID_JSON_PENALTY = -0.5
TRAIN_STEP_ERROR_PENALTY = -0.1
