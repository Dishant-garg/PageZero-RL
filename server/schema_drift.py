import random

from .config import (
    DRIFT_MIN_DIFFICULTY_DB, DRIFT_MIN_DIFFICULTY_CACHE,
    DRIFT_PROBABILITY, DRIFT_TRIGGER_STEP_DB, DRIFT_TRIGGER_STEP_CACHE,
)


class SchemaDriftEngine:
    """Injects mutations into the environment mid-episode to force the agent to adapt.

    Each drift event carries a ``reverse_command`` so the environment can
    undo it during reset — preventing permanent corruption across episodes.
    """

    def __init__(self):
        self.drift_events = [
            {
                "type": "column_rename",
                "description": "Database column 'user_email' renamed to 'email_address'",
                "command": "ALTER TABLE orders RENAME COLUMN user_email TO email_address;",
                "reverse_command": "ALTER TABLE orders RENAME COLUMN email_address TO user_email;",
                "trigger_step": DRIFT_TRIGGER_STEP_DB,
                "min_difficulty": DRIFT_MIN_DIFFICULTY_DB,
                "layer": "database",
            },
            {
                "type": "redis_key_format",
                "description": "Redis keys changed from 'daily_stats' to 'stats:daily:v2'",
                "command": "RENAME daily_stats stats:daily:v2",
                "reverse_command": None,  # Redis keys are cleaned by FLUSHDB on reset
                "trigger_step": DRIFT_TRIGGER_STEP_CACHE,
                "min_difficulty": DRIFT_MIN_DIFFICULTY_CACHE,
                "layer": "cache",
                # Guard: only fire if the key actually exists
                "precondition_cmd": "EXISTS daily_stats",
            },
        ]
        self.has_drifted = False
        self.applied_drift = None  # track which drift was applied

    def maybe_drift(self, current_step: int, difficulty: float) -> dict | None:
        if self.has_drifted:
            return None

        possible = [
            d for d in self.drift_events
            if d["trigger_step"] == current_step and difficulty >= d["min_difficulty"]
        ]

        if possible and random.random() < DRIFT_PROBABILITY:
            event = random.choice(possible)
            self.has_drifted = True
            self.applied_drift = event
            return event

        return None

    def get_applied_drift(self) -> dict | None:
        """Return the drift that was applied this episode (for reversal on reset)."""
        return self.applied_drift

    def reset(self):
        """Reset drift tracking for a new episode."""
        self.has_drifted = False
        self.applied_drift = None
