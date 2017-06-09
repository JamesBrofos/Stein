import numpy as np
from .abstract_gradient_descent import AbstractGradientDescent


class Adam(AbstractGradientDescent):
    """Adam Gradient Descent Class"""
    def __init__(self, learning_rate=1e-3, beta_1=0.9, beta_2=0.999):
        """Initialize the parameters of the Adam gradient descent object."""
        super(Adam, self).__init__(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def update(self, phi):
        """Implementation of abstract base class method."""
        # Keep an exponentially decaying average of previous gradients and
        # squared gradients.
        if self.n_iters == 0:
            self.mu, self.nu = phi, phi**2
        else:
            self.mu = self.beta_1 * self.mu + (1.-self.beta_1) * phi
            self.nu = self.beta_2 * self.nu + (1.-self.beta_2) * phi**2

        # Update the number of iterations.
        self.n_iters += 1
        mup = self.mu / (1. - self.beta_1**self.n_iters)
        nup = self.nu / (1. - self.beta_2**self.n_iters)

        return mup / (1e-8 + np.sqrt(nup)) * self.learning_rate
