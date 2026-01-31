import numpy as np


class SparseGridEnv:
    ACTION_TO_DELTA = {
        0: (-1, 0),
        1: (0, 1),
        2: (1, 0),
        3: (0, -1),
    }

    def __init__(self, size=6, max_steps=25, seed=0):
        self.size = size
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.start = (size - 1, 0)
        self.goal = (0, size - 1)
        self.state = None
        self.steps = 0

    @property
    def n_states(self):
        return self.size * self.size

    @property
    def n_actions(self):
        return 4

    def reset(self):
        self.state = self.start
        self.steps = 0
        return self._state_to_index(self.state)

    def step(self, action):
        self.steps += 1
        row, col = self.state
        dr, dc = self.ACTION_TO_DELTA[action]
        next_row = int(np.clip(row + dr, 0, self.size - 1))
        next_col = int(np.clip(col + dc, 0, self.size - 1))
        self.state = (next_row, next_col)

        reward = 0.0
        done = False
        if self.state == self.goal:
            reward = 1.0
            done = True
        elif self.steps >= self.max_steps:
            done = True

        return self._state_to_index(self.state), reward, done, {}

    def _state_to_index(self, state):
        row, col = state
        return row * self.size + col
