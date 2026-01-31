import numpy as np


class TwoStepEnv:
    """
    State 0 -> choose A or B -> state 1 or 2 -> choose A or B -> terminal.
    Only the terminal reward matters (delayed).
    """

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.state = 0

    @property
    def n_states(self):
        return 3

    @property
    def n_actions(self):
        return 2

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        if self.state == 0:
            self.state = 1 if action == 0 else 2
            return self.state, 0.0, False, {}

        reward = 3.0 if (self.state == 1 and action == 0) else 1.0
        done = True
        return self.state, reward, done, {}
