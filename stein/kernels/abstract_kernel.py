import numpy as np
from abc import ABCMeta, abstractmethod
from scipy.spatial.distance import cdist


class AbstractKernel(object):
    """Abstract Kernel Class

    This class implements the template functionalities of a kernel for use with
    the Stein library. A kernel allows us to project inputs into a higher
    dimensional (and oftentimes infinite-dimensional) space where distances can
    be usefully expressed. This class implements a method for computing the
    squared Euclidean distance between points in the input space, as well as an
    abstract method for computing the kernel and gradient of the kernel matrix.
    """
    def squared_distance_and_bandwidth(self, theta):
        """Compute the pairwise squared Euclidean distances between all of the
        particles. This function also computes the bandwidth to use in the
        kernel, defaulting to a median-based heuristic if no bandwidth is
        supplied by the user.

        Parameters:
            theta (numpy array): A matrix representation of the particles and
                values assumed by each of the parameters in the model. The
                dimensions of the matrix are the number of particles by the
                number of parameters.

        Returns:
            Tuple: A tuple consisting of the squared Euclidean distances between
                a row of `theta` and all of the other rows and a bandwidth
                parameter that is computed using a heuristic which asserts that
                the "contribution of a point's own gradient and the influence of
                all other points balance with each other" (see the Stein
                variational gradient descent paper).
        """
        sq_dists = cdist(theta, theta, metric="sqeuclidean")
        h = np.sqrt(0.5 * np.median(sq_dists) / np.log(theta.shape[0] + 1))
        return sq_dists, h

    @abstractmethod
    def kernel_and_grad(self, theta):
        """Computes both the kernel matrix for the given input as well as the
        gradient of the kernel with respect to the first input, averaged over
        inputs.

        Parameters:
            theta (numpy array): A matrix representation of the particles and
                values assumed by each of the parameters in the model. The
                dimensions of the matrix are the number of particles by the
                number of parameters.

        Returns:
            Tuple: A tuple consisting of the kernel matrix computed from the
                rows of `theta` and the gradient of the kernel matrix with
                respect to each of the particles.
        """
        raise NotImplementedError()
