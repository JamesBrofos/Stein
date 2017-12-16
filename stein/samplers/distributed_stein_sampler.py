import numpy as np
import tensorflow as tf
import threading as th
from .abstract_stein_sampler import AbstractSteinSampler


class DistributedSteinSampler(AbstractSteinSampler):
    """Distributed Stein Sampler Class

    """
    def __init__(
            self, n_threads, n_particles, n_shuffle, log_p, gd, theta=None
    ):
        """Initialize the parameters of the distributed Stein sampler class.

        """
        # Number of threads, particles, and iterations to perform before
        # shuffling the particles amongst threads.
        self.n_threads = n_threads
        self.n_particles = n_particles
        self.n_shuffle = n_shuffle
        # Number of particles that each worker should train with.
        self.particles_per_worker = self.n_particles // self.n_threads
        # Create the overarching TensorFlow session.
        self.sess = tf.Session()

    def train_on_batch(self, batch_feed):
        """Implementation of abstract base class method."""


