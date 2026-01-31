from pathlib import Path

import numpy as np

from env import TwoStepEnv


def softmax(logits):
    exp = np.exp(logits - np.max(logits))
    return exp / np.sum(exp)


def train(episodes=400, learning_rate=0.1, seed=0):
    env = TwoStepEnv(seed=seed)
    rng = np.random.default_rng(seed)

    policy_logits = np.zeros((env.n_states, env.n_actions), dtype=np.float32)
    baseline = 0.0
    baseline_alpha = 0.1
    history = []

    for _ in range(episodes):
        state = env.reset()
        states = []
        actions = []
        rewards = []

        done = False
        while not done:
            probs = softmax(policy_logits[state])
            action = int(rng.choice(env.n_actions, p=probs))
            next_state, reward, done, _ = env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state

        G = sum(rewards)
        baseline = (1 - baseline_alpha) * baseline + baseline_alpha * G
        advantage = G - baseline

        for s, a in zip(states, actions):
            probs = softmax(policy_logits[s])
            grad = -probs
            grad[a] += 1.0
            policy_logits[s] += learning_rate * advantage * grad

        history.append(softmax(policy_logits[0])[0])

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    np.save(output_dir / "policy_probs.npy", np.array(history, dtype=np.float32))


if __name__ == "__main__":
    train()
