import random

from .config import (
    STARTING_DIFFICULTY, DIFFICULTY_STEP, MAX_DIFFICULTY,
    SUCCESS_THRESHOLD, MIN_EPISODES_TO_ADVANCE,
    WARMUP_PROB_LOW, WARMUP_PROB_MID, WARMUP_PROB_HIGH,
    WARMUP_THRESHOLD_LOW, WARMUP_THRESHOLD_HIGH,
)


class Curriculum:
    """Adaptive difficulty curriculum for RL training.

    Tracks per-layer skill scores and advances difficulty when the agent
    consistently succeeds.  Also provides ``get_weakest_layer()`` so the
    scenario designer can target the agent's weak spots.
    """

    def __init__(self):
        self.skill_profile = {
            "database": [],
            "cache": [],
            "application": [],
            "infrastructure": [],
            "cross_layer": [],
        }
        self.difficulty = STARTING_DIFFICULTY
        self.success_threshold = SUCCESS_THRESHOLD
        self.episodes_completed = 0

    def get_difficulty(self) -> float:
        return self.difficulty

    def record_result(self, scenario_layer: str, score: float):
        """Record the result of an episode and possibly advance difficulty."""
        if scenario_layer in self.skill_profile:
            self.skill_profile[scenario_layer].append(score)
        self.episodes_completed += 1

        # Calculate average of recent N runs in that layer
        if scenario_layer in self.skill_profile:
            recent = self.skill_profile[scenario_layer][-MIN_EPISODES_TO_ADVANCE:]
            if len(recent) >= MIN_EPISODES_TO_ADVANCE:
                avg_score = sum(recent) / len(recent)
                if avg_score > self.success_threshold:
                    self.difficulty = min(MAX_DIFFICULTY, self.difficulty + DIFFICULTY_STEP)

    def get_weakest_layer(self) -> str:
        """Return the layer with the lowest average score (for targeted training)."""
        averages = {}
        for layer, scores in self.skill_profile.items():
            if scores:
                averages[layer] = sum(scores[-10:]) / len(scores[-10:])
            else:
                averages[layer] = 0.0
        return min(averages, key=averages.get)

    def should_use_warmup(self) -> bool:
        """Higher difficulty → less warmup (more LLM-generated / hard scenarios)."""
        if self.difficulty < WARMUP_THRESHOLD_LOW:
            return random.random() < WARMUP_PROB_LOW
        elif self.difficulty < WARMUP_THRESHOLD_HIGH:
            return random.random() < WARMUP_PROB_MID
        else:
            return random.random() < WARMUP_PROB_HIGH
