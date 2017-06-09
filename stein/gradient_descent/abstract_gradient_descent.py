from abc import ABCMeta, abstractmethod


class AbstractGradientDescent(object):
    """Abstract Gradient Descent Class"""
    def __init__(self, learning_rate):
        """Initialize the parameters of the abstract gradient descent object."""
        self.learning_rate = learning_rate
        self.n_iters = 0

    @abstractmethod
    def update(self, phi):
        """Compute the update direction for the gradient descent algorithm."""
        raise NotImplementedError()

