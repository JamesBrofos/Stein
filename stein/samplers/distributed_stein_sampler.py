import numpy as np
import tensorflow as tf
import threading as th
from .abstract_stein_sampler import AbstractSteinSampler
from .stein_sampler import SteinSampler


class DistributedSteinSampler(AbstractSteinSampler):
    """Distributed Stein Sampler Class

    """
    def __init__(
            self, n_threads, n_particles, log_p, gd, theta=None
    ):
        """Initialize the parameters of the distributed Stein sampler class.

        """
        # Number of threads and particles.
        self.n_threads = n_threads
        self.n_particles = n_particles
        # Keep track of the number of training iterations that have been
        # performed globally.
        self.n_iter = 0
        # Number of particles that each worker should train with.
        self.particles_per_worker = self.n_particles // self.n_threads

        # Create the dictionary of samples.
        self.theta = {
            v: np.random.normal(
                size=[self.n_particles] + v.get_shape().as_list()
            ) * 0.01
            for v in tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, "model"
            )
        }

        # Create the workers.
        self.workers = []
        for i in range(self.n_threads):
            self.workers.append(SteinSampler(
                self.particles_per_worker,
                log_p,
                gd
            ))

    def __train_on_batch(self, batch_feed, thread_index):
        self.workers[thread_index].train_on_batch(batch_feed)

    def train_on_batch(self, batch_feed):
        """Implementation of abstract base class method."""
        # Bundles.
        # Increment the number of training iterations.
        self.n_iter += 1
        # Shuffle the first dimension of the particles to avoid collapse to
        # identical samples.
        shuffle_idx = np.random.permutation(self.n_particles)
        self.theta = {v: x[shuffle_idx] for v, x in self.theta.items()}
        # Create the threads and append them to a list that can be later joined.
        threads = []
        for i in range(self.n_threads):
            # Assign the worker a given subset of the particles to update.
            self.workers[i].theta = {
                v: x[i:i+self.particles_per_worker]
                for v, x in self.theta.items()
            }
            # Create and start the thread.
            threads.append(th.Thread(
                target=self.__train_on_batch,
                args=(batch_feed, i)
            ))
            threads[i].start()

        # Block the main program to ensure that all threads complete before
        # proceeding.
        for t in threads:
            t.join()

        # Now copy the updated parameters back to the global dictionary of
        # particles.
        for v in self.theta:
            for i in range(self.n_threads):
                self.theta[v][i:i+self.particles_per_worker] = (
                    self.workers[i].theta[v]
                )

