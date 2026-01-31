import numpy as np


class LoggedBandit:
    def __init__(self, probabilities, seed=0):
        self.probabilities = np.array(probabilities, dtype=np.float32)
        self.rng = np.random.default_rng(seed)

    @property
    def n_actions(self):
        return len(self.probabilities)

    def step(self, action):
        reward = 1.0 if self.rng.random() < self.probabilities[action] else 0.0
        return reward
