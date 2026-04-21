import random

class Curriculum:
    def __init__(self):
        self.skill_profile = {
            "database": [],
            "cache": [],
            "application": [],
            "infrastructure": [],
            "cross_layer": [],
        }
        self.difficulty = 0.15
        self.success_threshold = 0.6
        self.episodes_completed = 0

    def get_difficulty(self) -> float:
        return self.difficulty

    def record_result(self, scenario_layer: str, score: float):
        if scenario_layer in self.skill_profile:
            self.skill_profile[scenario_layer].append(score)
        self.episodes_completed += 1

        # Calculate average of recent 5 runs in that layer
        recent = self.skill_profile[scenario_layer][-5:]
        if len(recent) >= 5:
            avg_score = sum(recent) / len(recent)
            if avg_score > self.success_threshold:
                self.difficulty = min(1.0, self.difficulty + 0.1)
                
    def get_weakest_layer(self) -> str:
        averages = {}
        for layer, scores in self.skill_profile.items():
            if scores:
                averages[layer] = sum(scores[-10:]) / len(scores[-10:])
            else:
                averages[layer] = 0.0
        return min(averages, key=averages.get)

    def should_use_warmup(self) -> bool:
        if self.difficulty < 0.4:
            return random.random() < 0.7  # 70% warmup
        elif self.difficulty < 0.7:
            return random.random() < 0.3  # 30% warmup
        else:
            return random.random() < 0.1  # 10% warmup
