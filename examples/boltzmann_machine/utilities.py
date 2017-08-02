import itertools
import numpy as np


def construct_pairwise_interactions(X):
    n_samples, n = X.shape
    p = (n * (n - 1)) // 2
    B = np.zeros((n_samples, p))
    index = 0
    for i in range(1, n):
        for j in range(i):
            B[:, index] = X[:, i] * X[:, j]
            index += 1

    return B.astype(np.float32)

def enumerate_binary(n):
    # Create a vector to store the enumeration of the space.
    B = list(itertools.product([0, 1], repeat=n))
    return np.array(B).astype(np.float32)
