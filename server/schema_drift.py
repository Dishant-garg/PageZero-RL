import random

class SchemaDriftEngine:
    """Injects mutations into the environment mid-episode to force the agent to adapt."""
    
    def __init__(self):
        self.drift_events = [
            {
                "type": "column_rename",
                "description": "Database column 'user_email' renamed to 'email_address'",
                "command": "ALTER TABLE orders RENAME COLUMN user_email TO email_address;",
                "trigger_step": 5,
                "min_difficulty": 0.5,
                "layer": "database"
            },
            {
                "type": "redis_key_format",
                "description": "Redis keys changed from 'daily_stats' to 'stats:daily:v2'",
                "command": "RENAME daily_stats stats:daily:v2",
                "trigger_step": 6,
                "min_difficulty": 0.6,
                "layer": "cache"
            }
        ]
        self.has_drifted = False

    def maybe_drift(self, current_step: int, difficulty: float) -> dict:
        if self.has_drifted:
            return None
            
        possible = [
            d for d in self.drift_events 
            if d["trigger_step"] == current_step and difficulty >= d["min_difficulty"]
        ]
        
        if possible and random.random() < 0.5:  # 50% chance to happen if conditions met
            event = random.choice(possible)
            self.has_drifted = True
            return event
            
        return None
