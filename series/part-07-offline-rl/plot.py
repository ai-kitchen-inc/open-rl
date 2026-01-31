import matplotlib.pyplot as plt
import numpy as np


def main():
    q_values = np.load("outputs/estimated_values.npy")
    target_value = float(np.load("outputs/target_value.npy")[0])

    plt.figure(figsize=(5, 4))
    plt.bar(range(len(q_values)), q_values)
    plt.axhline(target_value, color="red", linestyle="--", label="Target policy value")
    plt.title("Offline estimated action values")
    plt.xlabel("Action")
    plt.ylabel("Estimated reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/offline_eval.png")
    plt.close()


if __name__ == "__main__":
    main()
