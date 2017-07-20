import numpy as np
import tensorflow as tf
from mpi4py import MPI
from .abstract_stein_sampler import AbstractSteinSampler
from .stein_sampler import SteinSampler
from ..utilities import convert_dictionary_to_array, convert_array_to_dictionary


class ParallelSteinSampler(AbstractSteinSampler):
    """Parallel Stein Sampler Class
    """
    def __init__(self, n_particles, log_p, gd, theta=None):
        """Initialize the parameters of the parallel Stein sampler object.
        """
        # Use MPI for communication between parallel samplers.
        self.comm = MPI.COMM_WORLD
        self.n_particles = n_particles
        self.n_workers = self.comm.size
        self.particles_per_worker = n_particles // self.n_workers
        if self.n_particles % self.n_workers != 0 and self.comm.rank == 0:
            raise ValueError(
                "The number of particles must be divisible by the number of "
                "worker processes."
            )

        # Partition the particles among the worker processes. If there was no
        # provided set of initial particles, then we can just initialize the
        # sampler via the default constructor.
        if theta is None:
            self.sampler = SteinSampler(self.particles_per_worker, log_p, gd)
        else:
            idx = self.comm.rank*self.particles_per_worker
            self.sampler = SteinSampler(
                self.particles_per_worker,
                log_p,
                gd,
                {
                    v: x[idx:idx+self.particles_per_worker]
                    for v, x in theta.items()
                }
            )

    def train_on_batch(self, batch_feed):
        """Implementation of abstract base class method."""
        self.sampler.train_on_batch(batch_feed)

    def merge(self):
        """This method assembles the particles from all of the worker processes
        and returns a dictionary mapping TensorFlow variables to the
        corresponding value of each particle.

        Returns:
            Dict: A dictionary mapping TensorFlow variables to matrices where
                each row is a particle and each column is a parameter for that
                variable.
        """
        # Every process converts its TensorFlow dictionary of parameters into a
        # numpy array and saves the indices for how to reconstruct the
        # dictionary.
        theta_array, access = convert_dictionary_to_array(self.sampler.theta)
        if self.comm.rank == 0:
            # The master process requests each worker to send it a numpy
            # representation of its particles. The concatenated numpy
            # representation is then mapped back to a dictionary.
            for i in range(1, self.n_workers):
                theta_array = np.vstack((
                    theta_array, self.comm.recv(source=MPI.ANY_SOURCE)
                ))
            return convert_array_to_dictionary(theta_array, access)
        else:
            # Every worker process sends the master process a numpy
            # representation of its particles.
            self.comm.send(theta_array, dest=0)

    def shuffle(self):
        """This method shuffles the particles amongst the processes. This allows
        us to enforce the idea that each worker's particles should not collapse
        to the same sample. The idea is to enforce diversity by periodically
        transmitting particles to randomly selected destination processes, where
        the repulsive effect of the kernel will be different.

        TODO: Do we need to clear out the gradient descent parameters? Namely,
        historical gradients and historical squared gradients?
        """
        # Merge together all the particles. Notice that for all processes except
        # the master node this returns none.
        theta = self.merge()

        if self.comm.rank == 0:
            # Create an assignment of the destination for each particle.
            a = np.random.permutation(self.n_particles)
            assign = np.reshape(a, (self.n_workers, self.particles_per_worker))
            # Create a big array of all the particles.
            theta_array, access = convert_dictionary_to_array(theta)
            for i in range(1, self.n_workers):
                self.comm.send(theta_array[assign[i]], dest=i)
            theta_array = theta_array[assign[0]]
        else:
            # Initialize.
            _, access = convert_dictionary_to_array(self.sampler.theta)
            # Receive the destinations for the particles from the master process.
            theta_array = self.comm.recv(source=0)

        # Convert the numpy array back to a dictionary using the saved access
        # indices.
        self.sampler.theta = convert_array_to_dictionary(theta_array, access)
