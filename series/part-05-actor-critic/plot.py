import matplotlib.pyplot as plt
import numpy as np


def main():
    rewards = np.load("outputs/rewards.npy")
    window = 20
    smooth = np.convolve(rewards, np.ones(window) / window, mode="valid")

    plt.figure(figsize=(6, 4))
    plt.plot(rewards, alpha=0.4)
    plt.plot(np.arange(len(smooth)) + (window - 1), smooth)
    plt.title("Actor-critic learning curve")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig("outputs/learning_curve.png")
    plt.close()


if __name__ == "__main__":
    main()
