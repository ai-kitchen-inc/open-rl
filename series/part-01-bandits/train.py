from pathlib import Path

import numpy as np

from env import BanditEnv


def train(steps=500, epsilon=0.1, seed=0):
    env = BanditEnv([0.2, 0.5, 0.8], seed=seed)
    rng = np.random.default_rng(seed)

    q_values = np.zeros(env.n_actions, dtype=np.float32)
    counts = np.zeros(env.n_actions, dtype=np.int32)
    rewards = []

    for _ in range(steps):
        if rng.random() < epsilon:
            action = int(rng.integers(env.n_actions))
        else:
            action = int(np.argmax(q_values))

        reward = env.step(action)
        counts[action] += 1
        q_values[action] += (reward - q_values[action]) / counts[action]
        rewards.append(reward)

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    np.save(output_dir / "rewards.npy", np.array(rewards, dtype=np.float32))


if __name__ == "__main__":
    train()
