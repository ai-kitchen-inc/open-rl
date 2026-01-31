import numpy as np


class QLearningAgent:
    def __init__(
        self,
        n_states,
        n_actions,
        learning_rate=0.2,
        discount=0.95,
        epsilon=0.2,
        seed=7,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

        self.q_table = np.zeros((n_states, n_actions), dtype=np.float32)

    def select_action(self, state):
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state, done):
        best_next = np.max(self.q_table[next_state])
        target = reward + (0.0 if done else self.discount * best_next)
        td_error = target - self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * td_error

    def greedy_action(self, state):
        return int(np.argmax(self.q_table[state]))
