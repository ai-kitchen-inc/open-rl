import argparse
import json
from pathlib import Path

import numpy as np

from environment import GridworldEnv
from agent import QLearningAgent


def run_episode(env, agent, explore=True):
    state = env.reset()
    total_reward = 0.0
    done = False

    while not done:
        action = agent.select_action(state) if explore else agent.greedy_action(state)
        next_state, reward, done, _ = env.step(action)
        if explore:
            agent.update(state, action, reward, next_state, done)
        total_reward += reward
        state = next_state

    return total_reward


def train(args):
    env = GridworldEnv(size=5, max_steps=30, seed=args.seed)
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        learning_rate=args.learning_rate,
        discount=args.discount,
        epsilon=args.epsilon,
        seed=args.seed,
    )

    rewards = []
    for _ in range(args.episodes):
        episode_reward = run_episode(env, agent, explore=True)
        rewards.append(episode_reward)

    eval_rewards = [run_episode(env, agent, explore=False) for _ in range(20)]
    results = {
        "train_rewards": rewards,
        "eval_rewards": eval_rewards,
        "avg_eval_reward": float(np.mean(eval_rewards)),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "train_rewards.npy", np.array(rewards, dtype=np.float32))
    np.save(output_dir / "q_table.npy", agent.q_table)

    with (output_dir / "results.json").open("w") as handle:
        json.dump(results, handle, indent=2)

    print(json.dumps(results, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Q-learning agent.")
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.2)
    parser.add_argument("--discount", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=str, default="outputs")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
