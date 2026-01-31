from pathlib import Path

import numpy as np

from env import SafeGridEnv


def train(episodes=300, penalty=5.0, seed=0):
    env = SafeGridEnv(seed=seed)
    rng = np.random.default_rng(seed)
    q_table = np.zeros((env.n_states, env.n_actions), dtype=np.float32)

    rewards = []
    violations = []

    for _ in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        total_violations = 0
        while not done:
            if rng.random() < 0.2:
                action = int(rng.integers(env.n_actions))
            else:
                action = int(np.argmax(q_table[state]))
            next_state, reward, done, info = env.step(action)
            safety_penalty = penalty * info["violation"]
            adjusted_reward = reward - safety_penalty
            q_table[state, action] += 0.2 * (
                adjusted_reward + 0.95 * np.max(q_table[next_state]) - q_table[state, action]
            )

            total_reward += reward
            total_violations += info["violation"]
            state = next_state

        rewards.append(total_reward)
        violations.append(total_violations)

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    np.save(output_dir / "rewards.npy", np.array(rewards, dtype=np.float32))
    np.save(output_dir / "violations.npy", np.array(violations, dtype=np.int32))


if __name__ == "__main__":
    train()
