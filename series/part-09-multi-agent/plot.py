import matplotlib.pyplot as plt
import numpy as np


def main():
    counts = np.load("outputs/joint_counts.npy")
    labels = ["(0,0)", "(0,1)", "(1,0)", "(1,1)"]
    values = counts.flatten()

    plt.figure(figsize=(5, 4))
    plt.bar(labels, values)
    plt.title("Joint action frequency")
    plt.xlabel("Joint action")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("outputs/joint_actions.png")
    plt.close()


if __name__ == "__main__":
    main()
