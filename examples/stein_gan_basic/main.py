import numpy as np
import matplotlib.pyplot as plt
from generate_samples import generate_samples


# Draw samples from the target distribution.
n_samples = 10000
X = generate_samples(n_samples)

if True:
    plt.figure(figsize=(8, 6))
    plt.hist(X, bins=50, normed=True)
    plt.grid()
    plt.show()
