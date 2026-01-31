import matplotlib.pyplot as plt
import numpy as np

from env import SparseGridEnv


def main():
    env = SparseGridEnv()
    counts = np.load("outputs/visit_counts.npy").reshape(env.size, env.size)

    plt.figure(figsize=(5, 4))
    plt.imshow(counts, cmap="viridis")
    plt.colorbar(label="Visits")
    plt.title("Exploration coverage")
    plt.tight_layout()
    plt.savefig("outputs/visits_heatmap.png")
    plt.close()


if __name__ == "__main__":
    main()
