from pathlib import Path

import numpy as np

from env import SparseGridEnv


def train(episodes=200, bonus_scale=0.5, seed=0):
    env = SparseGridEnv(seed=seed)
    rng = np.random.default_rng(seed)
    q_table = np.zeros((env.n_states, env.n_actions), dtype=np.float32)
    visit_counts = np.zeros(env.n_states, dtype=np.int32)

    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            visit_counts[state] += 1
            bonus = bonus_scale / np.sqrt(visit_counts[state])
            if rng.random() < 0.1:
                action = int(rng.integers(env.n_actions))
            else:
                action = int(np.argmax(q_table[state] + bonus))
            next_state, reward, done, _ = env.step(action)
            q_table[state, action] += 0.2 * (reward + 0.95 * np.max(q_table[next_state]) - q_table[state, action])
            state = next_state

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    np.save(output_dir / "visit_counts.npy", visit_counts)


if __name__ == "__main__":
    train()
