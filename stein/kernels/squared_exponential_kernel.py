import tensorflow as tf
import numpy as np
from scipy.spatial.distance import cdist
from .abstract_kernel import AbstractKernel


class SquaredExponentialKernel(AbstractKernel):
    """Squared Exponential Kernel Class

    The squared exponential kernel (also called a Gaussian kernel or a radial
    basis function kernel) is implemented in this class. Notice that we use a
    constant bandwidth across every dimension of the input, yielding an
    isotropic kernel.

    The squared exponential kernel computes the squared Euclidean distances
    between particles, scales them by the squared bandwidth, and then
    exponentiates that value.
    """
    def __init__(self, n_particles, sess):
        """Initialize the parameters of the squared exponential kernel object.
        """
        super(SquaredExponentialKernel, self).__init__(n_particles, sess)
        self.K = tf.exp(-self.D / tf.square(self.bandwidth) / 2.)
        self.dK = tf.gradients(self.K, self.theta)

    def kernel_and_grad(self, theta):
        """Implementation of abstract base class method."""
        feed = {self.theta[i]: theta[i] for i in range(self.n_particles)}
        kernel, grads = self.sess.run([self.K, self.dK], feed)
        # TODO: Why do we need to multiply by 1/2 to get numbers consistent with
        # original SVGD paper?
        grads = -0.5 * np.vstack(grads)
        # grads = -np.vstack(grads)
        return kernel, grads
