from pathlib import Path

import numpy as np

from env import CoordinationGame


def train(episodes=400, epsilon=0.2, seed=0):
    env = CoordinationGame()
    rng = np.random.default_rng(seed)

    q_a = np.zeros(env.n_actions, dtype=np.float32)
    q_b = np.zeros(env.n_actions, dtype=np.float32)
    joint_counts = np.zeros((env.n_actions, env.n_actions), dtype=np.int32)

    for _ in range(episodes):
        action_a = int(rng.integers(env.n_actions)) if rng.random() < epsilon else int(np.argmax(q_a))
        action_b = int(rng.integers(env.n_actions)) if rng.random() < epsilon else int(np.argmax(q_b))

        reward = env.step(action_a, action_b)
        q_a[action_a] += 0.2 * (reward - q_a[action_a])
        q_b[action_b] += 0.2 * (reward - q_b[action_b])
        joint_counts[action_a, action_b] += 1

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    np.save(output_dir / "joint_counts.npy", joint_counts)


if __name__ == "__main__":
    train()
