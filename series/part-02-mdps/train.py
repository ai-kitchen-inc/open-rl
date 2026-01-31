from pathlib import Path

import numpy as np

from env import TwoRoomEnv


def run_episode(env, rng):
    state = env.reset()
    done = False
    trajectory = []
    while not done:
        action = int(rng.integers(4))
        next_state, reward, done, _ = env.step(action)
        trajectory.append((state, reward))
        state = next_state
    return trajectory


def monte_carlo_value(episodes=500, seed=0):
    env = TwoRoomEnv(seed=seed)
    rng = np.random.default_rng(seed)
    returns = {s: [] for s in range(env.n_states)}

    for _ in range(episodes):
        trajectory = run_episode(env, rng)
        G = 0.0
        for state, reward in reversed(trajectory):
            G += reward
            returns[state].append(G)

    values = np.zeros(env.n_states, dtype=np.float32)
    for state, samples in returns.items():
        values[state] = np.mean(samples) if samples else 0.0

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    np.save(output_dir / "state_values.npy", values)


if __name__ == "__main__":
    monte_carlo_value()
