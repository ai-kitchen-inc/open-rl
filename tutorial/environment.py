import numpy as np


class GridworldEnv:
    """
    A tiny gridworld with delayed reward.

    - The agent starts at the bottom-left corner.
    - The goal is the top-right corner.
    - Every step costs -1.
    - Reaching the goal gives +10 and ends the episode.
    - Falling into a trap gives -10 and ends the episode.
    """

    ACTIONS = ["up", "right", "down", "left"]
    ACTION_TO_DELTA = {
        0: (-1, 0),  # up
        1: (0, 1),   # right
        2: (1, 0),   # down
        3: (0, -1),  # left
    }

    def __init__(self, size=5, max_steps=30, seed=7):
        self.size = size
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        self.start = (size - 1, 0)
        self.goal = (0, size - 1)
        self.traps = {(1, 3), (2, 2)}

        self.state = None
        self.steps = 0

    @property
    def n_states(self):
        return self.size * self.size

    @property
    def n_actions(self):
        return len(self.ACTIONS)

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

        reward = -1.0
        done = False

        if self.state == self.goal:
            reward = 10.0
            done = True
        elif self.state in self.traps:
            reward = -10.0
            done = True
        elif self.steps >= self.max_steps:
            done = True

        return self._state_to_index(self.state), reward, done, {}

    def _state_to_index(self, state):
        row, col = state
        return row * self.size + col

    def index_to_state(self, index):
        row = index // self.size
        col = index % self.size
        return (row, col)
