import itertools
import numpy as np


def construct_pairwise_interactions(X):
    """Computes all of the pairwise products between the columns of the input
    matrix. This excludes self-interactions so that only products between
    non-equal column indices are considered.
    """
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
    """Enumerates all binary combinations of `n` bits. For instance, for an `n`
    equal to two, we would produce,

        [[0, 0], [0, 1], [1, 0], [0, 0]]

    as a numpy array.
    """
    B = list(itertools.product([0, 1], repeat=n))
    return np.array(B).astype(np.float32)
