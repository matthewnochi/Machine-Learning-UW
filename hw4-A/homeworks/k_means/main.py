if __name__ == "__main__":
    from k_means import lloyd_algorithm  # type: ignore
else:
    from .k_means import lloyd_algorithm

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem


@problem.tag("hw4-A", start_line=1)
def main():
    """Main function of k-means problem

    Run Lloyd's Algorithm for k=10, and report 10 centers returned.

    NOTE: This code might take a while to run. For debugging purposes you might want to change:
        x_train to x_train[:10000]. Make sure to change it back before submission!
    """
    (x_train, _), _ = load_dataset("mnist")

    num_centers = 10
    x_trainNew = x_train.reshape(x_train.shape[0], -1)
    centers, error = lloyd_algorithm(x_trainNew, num_centers=num_centers)

    centersNew = centers.reshape(num_centers, 28, 28)

    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i in range(num_centers):
        ax = axes[i // 5, i % 5]
        ax.imshow(centersNew[i], cmap='gray')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
