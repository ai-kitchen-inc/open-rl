import matplotlib.pyplot as plt
import numpy as np


def main():
    probs = np.load("outputs/policy_probs.npy")

    plt.figure(figsize=(6, 4))
    plt.plot(probs)
    plt.title("Policy probability of action A at state 0")
    plt.xlabel("Episode")
    plt.ylabel("P(action A)")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("outputs/policy_probs.png")
    plt.close()


if __name__ == "__main__":
    main()
