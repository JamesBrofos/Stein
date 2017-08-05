import numpy as np
import tensorflow as tf
from time import time
from .abstract_stein_sampler import AbstractSteinSampler
from ..utilities.converters import convert_dictionary_to_array


class SteinSampler(AbstractSteinSampler):
    """Stein Sampler Class

    This class implements a sequential version of the Stein variational gradient
    descent algorithm that does not exploit parallelism. This means that
    computation of the gradient is done sequentially and then a global optimal
    perturbation is computed and applied.
    """
    def train_on_batch(self, batch_feed):
        """Implementation of abstract base class method."""
        # Initialize a dictionary to store the gradient with respect to each
        # constituent parameter of the particle.
        grads = {
            v: np.zeros([self.n_particles] + v.get_shape().as_list())
            for v in self.model_vars
        }
        # Iterate over particles and compute the gradient.
        for i in range(self.n_particles):
            # Combine the parameter feed dictionary with the data feed
            # dictionary. Unlike previous versions, this uses backwards
            # compatible code.
            theta_feed = {v: self.theta[v][i] for v in self.model_vars}
            theta_feed.update(batch_feed)
            grad = self.sess.run(self.grad_log_p, theta_feed)
            # Update the parameters of the current particle.
            for v, g in zip(self.model_vars, grad):
                grads[v][i] = g

        # Apply the optimal perturbation direction.
        self.update_particles(convert_dictionary_to_array(grads)[0])

    def function_posterior(self, func, feed_dict):
        """Implementation of abstract base class method."""
        # Initialize a vector to store the value of the function for each particle.
        dist = np.zeros((self.n_particles, ))
        # Iterate over each particle and compute the value of the function for
        # that posterior sample.
        for i in range(self.n_particles):
            feed_dict.update({v: x[i] for v, x in self.theta.items()})
            dist[i] = self.sess.run(func, feed_dict)

        # Either return posterior samples of the input function.
        return dist
