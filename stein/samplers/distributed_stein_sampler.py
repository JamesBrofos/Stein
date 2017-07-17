import numpy as np
import tensorflow as tf
from mpi4py import MPI
from .abstract_stein_sampler import AbstractSteinSampler


class DistributedSteinSampler(AbstractSteinSampler):
    """Distributed Stein Sampler Class"""
    def __init__(self, n_particles, log_p, gd, theta=None):
        """Initialize the parameters of the distributed Stein sampler object.
        """
        # Use MPI for distributed computation.
        self.comm = MPI.COMM_WORLD
        # We assign the master process to have rank zero.
        if self.comm.rank == 0:
            super(DistributedSteinSampler, self).__init__(
                n_particles, log_p, gd, theta
            )
        else:
            self.model_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, "model"
            )
            self.grad_log_p = tf.gradients(log_p, self.model_vars)
            self.sess = tf.Session()

    def train_on_batch(self, batch_feed):
        """Implementation of abstract base class method."""
        if self.comm.rank == 0:
            # This is the master process. The master process is responsible for
            # coordinating the gradient computations and combining them to 
            # update the optimal perturbation direction.
            for i in range(self.n_particles):
                param_dict = {v.name: self.theta[v][i] for v in self.model_vars}
                self.comm.send(param_dict, dest=i+1)

            # Initialize a dictionary to store the gradient with respect to each
            # constituent parameter of the particle.
            grads = {
                v: np.zeros([self.n_particles] + v.get_shape().as_list())
                for v in self.model_vars
            }
            for i in range(self.n_particles):
                grad_dict = self.comm.recv(source=i+1)
                for v, g in grad_dict.items():
                    var = next((x for x in self.model_vars if x.name == v), None)
                    if var is None:
                        raise ValueError("Could not find variable.")
                    grads[var][i] = g

            # Apply the optimal perturbation direction.
            self.update_particles(grads)
        else:
            # This is the worker process. It is responsible for computing the
            # gradient of the posterior log-likelihood with respect to the model
            # parameters.
            data = self.comm.recv(source=0)
            theta_feed = {}
            for v in data:
                var = next((x for x in self.model_vars if x.name == v), None)
                if var is None:
                    raise ValueError("Could not find variable.")
                theta_feed[var] = data[v]
            # Save space by clearing out the transmitted dictionary.
            del data
            # Update the variable feed dictionary with the data placeholders.
            theta_feed.update(batch_feed)
            # Compute the gradient of the log-posterior with respect to the
            # model parameters.
            grad = self.sess.run(self.grad_log_p, theta_feed)
            grad_dict = {v.name: g for v, g in zip(self.model_vars, grad)}
            # Send the gradient back to the master node.
            self.comm.send(grad_dict, dest=0)
