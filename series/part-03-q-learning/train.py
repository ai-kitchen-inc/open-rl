from pathlib import Path

import numpy as np

from env import GridworldEnv


def train(episodes=300, epsilon=0.2, seed=0):
    env = GridworldEnv(seed=seed)
    rng = np.random.default_rng(seed)
    q_table = np.zeros((env.n_states, env.n_actions), dtype=np.float32)

    rewards = []
    for _ in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            if rng.random() < epsilon:
                action = int(rng.integers(env.n_actions))
            else:
                action = int(np.argmax(q_table[state]))
            next_state, reward, done, _ = env.step(action)
            best_next = np.max(q_table[next_state])
            q_table[state, action] += 0.2 * (reward + 0.95 * best_next - q_table[state, action])
            total_reward += reward
            state = next_state
        rewards.append(total_reward)

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    np.save(output_dir / "train_rewards.npy", np.array(rewards, dtype=np.float32))
    np.save(output_dir / "q_table.npy", q_table)


if __name__ == "__main__":
    train()
