from pathlib import Path

import numpy as np

from env import LoggedBandit


def collect_logs(env, behavior_probs, steps=1000, seed=0):
    rng = np.random.default_rng(seed)
    logs = []
    for _ in range(steps):
        action = int(rng.choice(env.n_actions, p=behavior_probs))
        reward = env.step(action)
        logs.append((action, reward))
    return logs


def fitted_q_evaluation(logs, n_actions):
    sums = np.zeros(n_actions, dtype=np.float32)
    counts = np.zeros(n_actions, dtype=np.int32)
    for action, reward in logs:
        sums[action] += reward
        counts[action] += 1
    q_values = np.where(counts > 0, sums / counts, 0.0)
    return q_values


def main():
    env = LoggedBandit([0.2, 0.4, 0.7], seed=0)
    behavior_probs = np.array([0.6, 0.3, 0.1], dtype=np.float32)
    logs = collect_logs(env, behavior_probs)

    q_values = fitted_q_evaluation(logs, env.n_actions)
    target_policy = np.array([0.1, 0.2, 0.7], dtype=np.float32)
    estimated_value = float(np.dot(target_policy, q_values))

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    np.save(output_dir / "estimated_values.npy", q_values)
    np.save(output_dir / "target_value.npy", np.array([estimated_value], dtype=np.float32))


if __name__ == "__main__":
    main()
