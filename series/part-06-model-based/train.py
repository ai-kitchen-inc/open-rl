from pathlib import Path

import numpy as np

from env import MazeEnv


def q_update(q_table, state, action, reward, next_state, done):
    best_next = np.max(q_table[next_state])
    target = reward + 0.95 * best_next * (0.0 if done else 1.0)
    q_table[state, action] += 0.2 * (target - q_table[state, action])


def run(episodes=200, planning_steps=0, seed=0):
    env = MazeEnv(seed=seed)
    rng = np.random.default_rng(seed)
    q_table = np.zeros((env.n_states, env.n_actions), dtype=np.float32)
    model = {}
    rewards = []

    for _ in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action = int(rng.integers(env.n_actions))
            next_state, reward, done, _ = env.step(action)
            q_update(q_table, state, action, reward, next_state, done)
            model[(state, action)] = (next_state, reward, done)

            for _ in range(planning_steps):
                (s, a), (s_next, r, d) = rng.choice(list(model.items()))
                q_update(q_table, s, a, r, s_next, d)

            total_reward += reward
            state = next_state
        rewards.append(total_reward)

    return rewards


def main():
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    model_free = run(planning_steps=0)
    dyna = run(planning_steps=10)

    np.save(output_dir / "model_free_rewards.npy", np.array(model_free, dtype=np.float32))
    np.save(output_dir / "dyna_rewards.npy", np.array(dyna, dtype=np.float32))


if __name__ == "__main__":
    main()
