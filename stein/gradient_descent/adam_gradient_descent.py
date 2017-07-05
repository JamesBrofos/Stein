import numpy as np
from .abstract_gradient_descent import AbstractGradientDescent


class AdamGradientDescent(AbstractGradientDescent):
    """Adam Gradient Descent Class

    This class implements adaptive moment estimation (Adam) for updating the
    particles in Stein variational gradient descent. Adam computes an individual
    "effective learning rate" (controlled, of course, by the global step size)
    for each parameter in the model. This is accomplished by computing an
    exponentially weighted average of previous gradients as well as squared
    gradients.
    """
    def __init__(self, learning_rate=1e-3, beta_1=0.9, beta_2=0.999):
        """Initialize the parameters of the Adam gradient descent object.

        Parameters:
            learning_rate (float): A global step size parameter used for every
                gradient descent class. This determines the global magnitude of
                a gradient descent step by scaling the gradient by this value.
                For simple models such as logistic regression, this value can be
                set relatively large (e.g. 1e-1), but must be small (e.g. 1e-4)
                for complex models such as neural networks.
            beta_1 (float, optional): The control parameter for how much to
                decay previous gradients when computing the weighted average of
                the current gradient and the previous gradients.
            beta_2 (float, optional): The control parameter for how much to
                decay previous squared gradients when computing the weighted
                average of the current squared gradient and the previous squared
                gradients.
        """
        super(AdamGradientDescent, self).__init__(learning_rate)
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
