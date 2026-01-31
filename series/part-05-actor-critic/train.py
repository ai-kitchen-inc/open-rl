from pathlib import Path

import numpy as np

from env import MiniGridEnv


def softmax(logits):
    exp = np.exp(logits - np.max(logits))
    return exp / np.sum(exp)


def train(episodes=300, actor_lr=0.1, critic_lr=0.2, seed=0):
    env = MiniGridEnv(seed=seed)
    rng = np.random.default_rng(seed)

    actor = np.zeros((env.n_states, env.n_actions), dtype=np.float32)
    critic = np.zeros(env.n_states, dtype=np.float32)
    rewards = []

    for _ in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            probs = softmax(actor[state])
            action = int(rng.choice(env.n_actions, p=probs))
            next_state, reward, done, _ = env.step(action)

            td_target = reward + 0.95 * critic[next_state] * (0.0 if done else 1.0)
            td_error = td_target - critic[state]

            critic[state] += critic_lr * td_error

            grad = -probs
            grad[action] += 1.0
            actor[state] += actor_lr * td_error * grad

            total_reward += reward
            state = next_state
        rewards.append(total_reward)

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    np.save(output_dir / "rewards.npy", np.array(rewards, dtype=np.float32))


if __name__ == "__main__":
    train()
