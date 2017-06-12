import numpy as np


def generate_samples(n_samples, loc=0., scale=1.):
    """Sample from a univariate Gaussian distribution with a specified mean and
    variance.
    """
    return np.random.normal(loc=loc, scale=scale, size=(n_samples, ))
