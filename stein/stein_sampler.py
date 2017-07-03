import numpy as np
import tensorflow as tf
from .kernels import SquaredExponentialKernel
from .utilities import convert_array_to_dictionary, convert_dictionary_to_array


class SteinSampler(object):
    """Stein Sampler Class"""
    def __init__(self, n_particles, log_p, gd, theta=None):
        """Initialize the parameters of the Stein sampler object."""
        self.n_particles = n_particles
        self.kernel = SquaredExponentialKernel()
        self.gd = gd
        self.log_p = log_p
        self.sess = tf.Session()
        self.model_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, "model"
        )
        self.grad_log_p = tf.gradients(self.log_p, self.model_vars)
        self.sess.run(tf.global_variables_initializer())
        if theta is not None:
            self.theta = theta
        else:
            self.theta = {
                v: np.random.normal(
                    size=[self.n_particles] + v.get_shape().as_list()
                )
                for v in self.model_vars
            }

    def compute_phi(self, theta_array, grads_array):
        """Assuming a reproducing kernel Hilbert space with associated kernel,
        this function computes the optimal perturbation in the particles under
        functions in the unit ball under the norm of the RKHS. This perturbation
        can be regarded as the direction that will maximally decrease the
        KL-divergence between the empirical distribution of the particles and
        the target distribution.
        """
        # Extract the number of particles and number of parameters.
        n_particles, n_params = grads_array.shape
        # Compute the kernel matrices and gradient with respect to the
        # particles.
        K, dK = self.kernel.kernel_and_grad(theta_array)

        return (K.dot(grads_array) + dK) / n_particles

    def train_on_batch(self, batch_feed):
        """"""
        grads = {
                v: np.zeros([self.n_particles] + v.get_shape().as_list())
                for v in self.model_vars            
        }
        for i in range(self.n_particles):
            # Check out this cool syntax for merging two dictionaries. :)
            theta_feed = {v: self.theta[v][i] for v in self.model_vars}
            grad = self.sess.run(self.grad_log_p, {**batch_feed, **theta_feed})
            for v, g in zip(self.model_vars, grad):
                grads[v][i] = g

        grads_array, access_indices = convert_dictionary_to_array(grads)
        theta_array, _ = convert_dictionary_to_array(self.theta)
        phi = self.compute_phi(theta_array, grads_array)
        theta_array += self.gd.update(phi)
        self.theta = convert_array_to_dictionary(theta_array, access_indices)

