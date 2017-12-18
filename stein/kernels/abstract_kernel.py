import numpy as np
import tensorflow as tf
from abc import abstractmethod
from ..utilities import compute_median


class AbstractKernel:
    """Abstract Kernel Class

    This class implements the template functionalities of a kernel for use with
    the Stein library. A kernel allows us to project inputs into a higher
    dimensional (and oftentimes infinite-dimensional) space where distances can
    be usefully expressed. This class implements a method for computing the
    squared Euclidean distance between points in the input space, as well as an
    abstract method for computing the kernel and gradient of the kernel matrix.
    """
    def __init__(self, n_particles, sess):
        """Initialize the parameters of the abstract kernel object.

        Parameters:
            n_particles (int): The number of particles to use in the algorithm.
                This is equivalently the number of samples to generate from the
                target distribution.
            sess (TensorFlow Session): A TensorFlow session in which to execute
                operations in the TensorFlow computational graph.
        """
        # Store the number of particles.
        self.n_particles = n_particles
        # Compute the squared distance between particles.
        self.theta = [
            tf.placeholder(tf.float32, [None]) for _ in range(self.n_particles)
        ]
        T = tf.stack(self.theta)
        r = tf.reshape(tf.reduce_sum(T*T, 1), [-1, 1])
        self.D = r + tf.transpose(r) - 2 * tf.matmul(T, tf.transpose(T))

        # Compute the kernel bandwidth.
        m = compute_median(self.D)
        # Prevent gradients from propagating backwards through the median.
        self.bandwidth = tf.stop_gradient(tf.sqrt(m / np.log(self.n_particles)))

        # Set the TensorFlow session.
        self.sess = sess

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


