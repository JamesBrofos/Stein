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
        # Use MPI for communication between parallel samplers.
        self.comm = MPI.COMM_WORLD
        self.n_workers = self.comm.size
        self.particles_per_worker = n_particles // self.n_workers
        if n_particles % self.n_workers != 0 and self.comm.rank == 0:
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
                n_particles,
                log_p,
                gd,
                theta[idx:idx + self.particles_per_worker]
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
        if self.comm.rank == 0:
            # Create an assignment of the destination for each particle.
            assign = np.tile(
                np.arange(self.comm.size), self.particles_per_worker
            )
            np.random.shuffle(assign)
            assign = assign.reshape((self.comm.size, self.particles_per_worker))
            # Broadcast the assignments to each node. Notice that we can skip
            # the node with rank zero.
            for i in range(1, self.n_workers):
                self.comm.send(assign[i], dest=i)

            assignments = assign[0]
        else:
            # Receive the destinations for the particles from the master process.
            assignments = self.comm.recv(source=0)

        # Every process converts its dictionary of parameters to a numpy array
        # and transmits the appropriate rows to the specified destination
        # processes
        theta_array, access = convert_dictionary_to_array(self.sampler.theta)
        for i in range(self.n_workers):
            if i == self.comm.rank:
                # If the destination is the process itself, no action is
                # required.
                continue
            # Create a boolean vector of which particles to transmit to each
            # process. Here we ensure that there is at least one particle to
            # transmit to the destination.
            send_idx = assignments == i
            if np.any(send_idx):
                self.comm.send(theta_array[send_idx], dest=i)
        # Filter out the transmitted particles and incrementally rebuild the
        # numpy array of particles by receiving from the other processes.
        theta_array = np.delete(
            theta_array, np.where(assignments != self.comm.rank)[0], axis=0
        )
        while theta_array.shape[0] < self.particles_per_worker:
            rec = self.comm.recv(source=MPI.ANY_SOURCE)
            theta_array = np.vstack((theta_array, rec))
        # Convert the numpy array back to a dictionary using the saved access
        # indices.
        self.sampler.theta = convert_array_to_dictionary(theta_array, access)
