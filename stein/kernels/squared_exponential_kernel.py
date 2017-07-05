import numpy as np
from .abstract_kernel import AbstractKernel


class SquaredExponentialKernel(AbstractKernel):
    """Squared Exponential Kernel Class"""

    def kernel_and_grad(self, theta):
        """Implementation of abstract base class method."""
        # Number of particles used to sample from the distribution.
        n_particles, n_params = theta.shape
        # Compute the pairwise squared Euclidean distances between all of the
        # particles and the bandwidth. This allows us to compute the kernel
        # matrix.
        sq_dists, h = self.squared_distance_and_bandwidth(theta)
        K = np.exp(-sq_dists / h**2 / 2.)
        # Compute the average of the gradient of the kernel with respect to
        # each of the particles.
        dK = np.zeros((n_particles, n_params))
        for i in range(n_particles):
            dK[i] = K[i].dot(theta[i] - theta) / (h**2)

        return K, dK
