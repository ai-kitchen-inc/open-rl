from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    rewards = np.load("outputs/rewards.npy")
    avg_rewards = np.cumsum(rewards) / (np.arange(len(rewards)) + 1)

    Path("outputs").mkdir(exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(avg_rewards)
    plt.title("Average reward over time")
    plt.xlabel("Step")
    plt.ylabel("Average reward")
    plt.tight_layout()
    plt.savefig("outputs/avg_reward.png")
    plt.close()


if __name__ == "__main__":
    main()
