import numpy as np
from .abstract_gradient_descent import AbstractGradientDescent


class Adagrad(AbstractGradientDescent):
    """Adagrad Gradient Descent Class"""
    def __init__(self, learning_rate=1e-3, alpha=0.9):
        """Initialize the parameters of the Adagrad gradient descent object."""
        super(Adagrad, self).__init__(learning_rate)
        self.alpha = alpha

    def update(self, phi):
        """Implementation of abstract base class method."""
        # Keep an exponentially decaying average of previous squared gradients.
        if self.n_iters == 0:
            self.hist = phi ** 2
        else:
            self.hist = self.alpha * self.hist + (1. - self.alpha) * phi ** 2
        # Update the number of iterations.
        self.n_iters += 1

        return phi / (1e-6 + np.sqrt(self.hist)) * self.learning_rate
