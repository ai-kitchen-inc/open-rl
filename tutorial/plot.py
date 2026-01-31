import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from environment import GridworldEnv


def plot_learning_curve(rewards, output_path):
    window = 20
    if len(rewards) >= window:
        smooth = np.convolve(rewards, np.ones(window) / window, mode="valid")
    else:
        smooth = rewards

    plt.figure(figsize=(7, 4))
    plt.plot(rewards, alpha=0.4, label="episode reward")
    plt.plot(np.arange(len(smooth)) + (window - 1), smooth, label="moving avg")
    plt.title("Learning curve")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_policy(q_table, output_path):
    env = GridworldEnv()
    policy = np.argmax(q_table, axis=1).reshape(env.size, env.size)

    arrow_map = {
        0: "↑",
        1: "→",
        2: "↓",
        3: "←",
    }

    plt.figure(figsize=(4, 4))
    plt.imshow(np.zeros((env.size, env.size)), cmap="Greys", vmin=0, vmax=1)
    for row in range(env.size):
        for col in range(env.size):
            symbol = arrow_map[policy[row, col]]
            if (row, col) == env.goal:
                symbol = "★"
            if (row, col) in env.traps:
                symbol = "✖"
            plt.text(col, row, symbol, ha="center", va="center", fontsize=12)
    plt.xticks([])
    plt.yticks([])
    plt.title("Greedy policy")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main(args):
    output_dir = Path(args.output_dir)
    rewards = np.load(output_dir / "train_rewards.npy")
    q_table = np.load(output_dir / "q_table.npy")

    plot_learning_curve(rewards, output_dir / "learning_curve.png")
    plot_policy(q_table, output_dir / "policy.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training outputs.")
    parser.add_argument("--output-dir", type=str, default="outputs")
    main(parser.parse_args())
