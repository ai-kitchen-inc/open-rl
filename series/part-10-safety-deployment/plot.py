import matplotlib.pyplot as plt
import numpy as np


def main():
    rewards = np.load("outputs/rewards.npy")
    violations = np.load("outputs/violations.npy")

    plt.figure(figsize=(6, 4))
    plt.plot(rewards, label="Reward")
    plt.plot(violations, label="Safety violations")
    plt.title("Reward vs. safety")
    plt.xlabel("Episode")
    plt.ylabel("Count / reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/safety_tradeoff.png")
    plt.close()


if __name__ == "__main__":
    main()
