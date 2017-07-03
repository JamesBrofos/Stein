import numpy as np
from abc import ABCMeta, abstractmethod
from scipy.spatial.distance import cdist


class AbstractKernel(object):
    """Abstract Kernel Class"""
    def __init__(self):
        """Initialize the parameters of the abstract kernel object."""

    def squared_distance_and_bandwidth(self, theta):
        """Compute the pairwise squared Euclidean distances between all of the
        particles. This function also computes the bandwidth to use in the
        kernel, defaulting to a median-based heuristic if no bandwidth is
        supplied by the user.
        """
        sq_dists = cdist(theta, theta, metric="sqeuclidean")
        h = np.sqrt(0.5 * np.median(sq_dists) / np.log(theta.shape[0] + 1))
        return sq_dists, h

    @abstractmethod
    def kernel_and_grad(self, theta):
        """Computes both the kernel matrix for the given input as well as the
        gradient of the kernel with respect to the first input, averaged over
        inputs.
        """
        raise NotImplementedError()
