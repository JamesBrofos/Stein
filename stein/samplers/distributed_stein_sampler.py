import numpy as np
import tensorflow as tf
from mpi4py import MPI
from .abstract_stein_sampler import AbstractSteinSampler
from ..utilities import convert_dictionary_to_array


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
            self.model_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, "model"
            )
            self.grad_log_p = tf.gradients(log_p, self.model_vars)
            self.sess = tf.Session()

        # Initialize a dictionary to store the gradient with respect to each
        # constituent parameter of the particle.
        self.n_params = int(sum([
            np.prod(v.get_shape().as_list()) for v in self.model_vars
        ]))
        self.n_workers = self.comm.size - 1
        self.particles_per_worker = n_particles // self.n_workers
        if n_particles % self.n_workers != 0:
            raise ValueError(
                "The number of particles must be divisible by the number of "
                "worker processes."
            )

    def train_on_batch(self, batch_feed):
        """Implementation of abstract base class method."""
        if self.comm.rank == 0:
            # This is the master process. The master process is responsible for
            # coordinating the gradient computations and combining them to
            # update the optimal perturbation direction.
            work = []
            work_indices = 0
            for i in range(self.n_workers):
                work.append((work_indices, work_indices + self.particles_per_worker))
                param_dict = {
                    v.name: self.theta[v][work[-1][0]:work[-1][1]]
                    for v in self.model_vars
                }
                work_indices += self.particles_per_worker
                self.comm.send(param_dict, dest=i+1)

            # Reassemble the matrix of gradients.
            G = np.empty((self.n_particles, self.n_params))
            for i in range(self.n_workers):
                s = MPI.Status()
                g = np.empty((self.particles_per_worker, self.n_params))
                self.comm.Recv([g, MPI.FLOAT], source=MPI.ANY_SOURCE, status=s)
                w = work[s.source-1]
                G[w[0]:w[1]] = g

            # Apply the optimal perturbation direction.
            self.update_particles(grads)
        else:
            # This is the worker process. It is responsible for computing the
            # gradient of the posterior log-likelihood with respect to the model
            # parameters.
            data = self.comm.recv(source=0)
            theta_feed = {}
            for i, v in enumerate(data):
                var = next((x for x in self.model_vars if x.name == v), None)
                theta_feed[var] = data[v]
            # Save space by clearing out the transmitted dictionary.
            del data
            # Initialize an empty array for the gradients for each particle
            # assigned to this worker process.
            G = np.empty((self.particles_per_worker, self.n_params))
            for i in range(self.particles_per_worker):
                batch_feed.update({v: theta_feed[v][i] for v in theta_feed})
                grad = self.sess.run(self.grad_log_p, batch_feed)
                grad_array = convert_dictionary_to_array(
                    {v: np.expand_dims(g, 0) for v, g in zip(self.model_vars, grad)}
                )[0].ravel()
                G[i] = grad_array

            # Send the gradient back to the master node.
            self.comm.Send([G, MPI.FLOAT], dest=0)
