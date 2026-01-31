import matplotlib.pyplot as plt
import numpy as np


def main():
    model_free = np.load("outputs/model_free_rewards.npy")
    dyna = np.load("outputs/dyna_rewards.npy")

    window = 10
    mf_smooth = np.convolve(model_free, np.ones(window) / window, mode="valid")
    dyna_smooth = np.convolve(dyna, np.ones(window) / window, mode="valid")

    plt.figure(figsize=(6, 4))
    plt.plot(mf_smooth, label="Model-free")
    plt.plot(dyna_smooth, label="Dyna-Q")
    plt.title("Planning improves learning speed")
    plt.xlabel("Episode")
    plt.ylabel("Smoothed reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/planning_comparison.png")
    plt.close()


if __name__ == "__main__":
    main()
