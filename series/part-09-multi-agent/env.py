class CoordinationGame:
    """
    Two agents choose action 0 or 1.
    Reward is higher when they coordinate on action 1 (stag).
    """

    def __init__(self):
        self.n_actions = 2

    def step(self, action_a, action_b):
        if action_a == 1 and action_b == 1:
            return 3.0
        if action_a == 0 and action_b == 0:
            return 1.0
        return 0.0
