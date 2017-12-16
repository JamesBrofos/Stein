import numpy as np
import tensorflow as tf
from .abstract_stein_sampler import AbstractSteinSampler
from ..kernels import SquaredExponentialKernel
from ..utilities.converters import convert_dictionary_to_array


class SteinSampler(AbstractSteinSampler):
    """Stein Sampler Class

    This class implements a sequential version of the Stein variational gradient
    descent algorithm that does not exploit parallelism. This means that
    computation of the gradient is done sequentially and then a global optimal
    perturbation is computed and applied.
    """
    def __init__(self, n_particles, log_p, gd, theta=None):
        """Initialize the parameters of the Stein sampler object.

        Parameters:
            n_particles (int): The number of particles to use in the algorithm.
                This is equivalently the number of samples to generate from the
                target distribution.
            log_p (TensorFlow tensor): A TensorFlow object corresponding to the
                log-posterior distribution from which parameters wish to be
                sampled. We only need to define the log-posterior up to an
                addative constant since we'll simply take the gradient with
                respect to the inputs and this term will vanish.
            gd (AbstractGradientDescent): An object that inherits from the
                abstract gradient descent object defined within the Stein
                library. This class is used to determine how to perturb the
                particles once the optimal perturbation direction. For instance,
                we might choose to update the particles according to the Adam
                optimizer scheme.
            theta (numpy array, optional): An optional parameter corresponding
                to the initial values of the particles. The dimension of this
                array (if it is provided) should be the number of particles by
                the number of random variables (parameters) to be sampled. If
                this value is not provided, then the initial particles will be
                generated by sampling from a multivariate standard normal
                distribution.
        """
        # Call the super class.
        super().__init__(n_particles, log_p, theta)
        # Gradient descent object will determine how particles are updated.
        self.gd = gd
        # Construct a squared exponential kernel for computing the repulsive
        # force between particles.
        self.kernel = SquaredExponentialKernel(self.n_particles, self.sess)

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

    @property
    def samples(self):
        """Converts the dictionary of sampled parameters into a numpy array for
        easier accessibility.
        """
        return convert_dictionary_to_array(self.theta)[0]

