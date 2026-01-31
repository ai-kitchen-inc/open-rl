import matplotlib.pyplot as plt
import numpy as np

from env import TwoRoomEnv


def main():
    env = TwoRoomEnv()
    values = np.load("outputs/state_values.npy").reshape(env.size, env.size)

    plt.figure(figsize=(4, 4))
    plt.imshow(values, cmap="coolwarm")
    plt.colorbar(label="State value")
    plt.title("Monte Carlo state values")
    plt.tight_layout()
    plt.savefig("outputs/state_values.png")
    plt.close()


if __name__ == "__main__":
    main()
