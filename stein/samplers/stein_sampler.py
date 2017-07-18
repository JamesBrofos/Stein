import numpy as np
import tensorflow as tf
from time import time
from .abstract_stein_sampler import AbstractSteinSampler


class SteinSampler(AbstractSteinSampler):
    """Stein Sampler Class

    This class implements a sequential version of the Stein variational gradient
    descent algorithm that does not exploit parallelism.
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
        self.update_particles(grads)
