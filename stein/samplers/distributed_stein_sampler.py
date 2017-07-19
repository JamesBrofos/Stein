import numpy as np
import tensorflow as tf
from mpi4py import MPI
from .abstract_stein_sampler import AbstractSteinSampler
from ..utilities import convert_dictionary_to_array, convert_array_to_dictionary


class DistributedSteinSampler(AbstractSteinSampler):
    """Distributed Stein Sampler Class"""
    def __init__(self, n_particles, log_p, gd, theta=None):
        """Initialize the parameters of the distributed Stein sampler object.
        """
        # Use MPI for distributed computation.
        self.comm = MPI.COMM_WORLD
        # We assign the master process to have rank zero.
        if self.comm.rank == 0:
            # Interpret the number of processes as the number of particles to
            # sample (this is done for simplicity). We subtract one because we
            # have a single master process.
            super(DistributedSteinSampler, self).__init__(
                n_particles, log_p, gd, theta
            )
        else:
            # TensorFlow setup on the worker processes.
            self.model_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, "model"
            )
            self.grad_log_p = tf.gradients(log_p, self.model_vars)
            self.sess = tf.Session()
            self.tf_variables = {v.name: v for v in self.model_vars}

        # Initialize a dictionary to store the gradient with respect to each
        # constituent parameter of the particle.
        self.n_params = int(sum([
            np.prod(v.get_shape().as_list()) for v in self.model_vars
        ]))
        self.n_workers = self.comm.size - 1
        self.particles_per_worker = n_particles // self.n_workers
        if n_particles % self.n_workers != 0 and self.comm.rank == 0:
            raise ValueError(
                "The number of particles must be divisible by the number of "
                "worker processes."
            )

    def master_process(self):
        """This is the master process. The master process is responsible for
        coordinating the gradient computations and combining them to update the
        optimal perturbation direction.
        """
        ppw = self.particles_per_worker
        work = [
            (i, i + ppw) for i in range(0, self.n_particles - ppw + 1, ppw)
        ]
        for i in range(self.n_workers):
            param_dict = {
                v.name: self.theta[v][work[i][0]:work[i][1]]
                for v in self.model_vars
            }
            self.comm.send(param_dict, dest=i+1)

        # Reassemble the matrix of gradients.
        G = np.empty((self.n_particles, self.n_params))
        for i in range(self.n_workers):
            # Initialize status for the receive and a buffer into which the
            # computed gradient will be loaded.
            s = MPI.Status()
            # Receive the gradient and place it into the corresponding rows of
            # the larger gradient matrix across all processes.
            g = self.comm.recv(source=MPI.ANY_SOURCE, status=s)
            w = work[s.source-1]
            G[w[0]:w[1]] = g

        # Apply the optimal perturbation direction.
        self.update_particles(G)

    def worker_process(self, batch_feed):
        """This is the worker process. It is responsible for computing the
        gradient of the posterior log-likelihood with respect to the model
        parameters.

        Parameters:
            batch_feed (dict): A dictionary that maps TensorFlow placeholders to
                provided values. For instance, this might be mappings of feature
                and target placeholders to batch values. Notice that this feed
                dictionary will be internally augmented to include the current
                feed values for the model parameters for each particle.
        """
        # Extract the transmitted dictionary into a dictionary with the same
        # value but with TensorFlow variable keys.
        data = self.comm.recv(source=0)
        theta_feed = {self.tf_variables[name]: data[name] for name in data}
        # Initialize an empty array for the gradients for each particle assigned
        # to this worker process.
        for i in range(self.particles_per_worker):
            batch_feed.update({v: theta_feed[v][i] for v in theta_feed})
            g = self.sess.run(self.grad_log_p, batch_feed)
            if i == 0:
                F = [np.expand_dims(h, 0) for h in g]
            else:
                F = [
                    np.vstack((F[i], np.expand_dims(h, 0)))
                    for i, h in enumerate(g)
                ]
            G, _ = convert_dictionary_to_array({
                v: g for v, g in zip(self.model_vars, F)
            })

        # Send the gradient back to the master node.
        self.comm.send(G, dest=0)

    def train_on_batch(self, batch_feed):
        """Implementation of abstract base class method."""
        if self.comm.rank == 0:
            self.master_process()
        else:
            self.worker_process(batch_feed)
