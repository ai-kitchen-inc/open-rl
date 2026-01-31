import matplotlib.pyplot as plt
import numpy as np

from env import GridworldEnv


def plot_policy(q_table, output_path):
    env = GridworldEnv()
    policy = np.argmax(q_table, axis=1).reshape(env.size, env.size)
    arrow_map = {0: "↑", 1: "→", 2: "↓", 3: "←"}

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
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    rewards = np.load("outputs/train_rewards.npy")
    q_table = np.load("outputs/q_table.npy")

    window = 20
    smooth = np.convolve(rewards, np.ones(window) / window, mode="valid")

    plt.figure(figsize=(6, 4))
    plt.plot(rewards, alpha=0.4)
    plt.plot(np.arange(len(smooth)) + (window - 1), smooth)
    plt.title("Learning curve")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig("outputs/learning_curve.png")
    plt.close()

    plot_policy(q_table, "outputs/policy.png")


if __name__ == "__main__":
    main()
