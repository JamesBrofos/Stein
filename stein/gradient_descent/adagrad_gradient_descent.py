import numpy as np
from .abstract_gradient_descent import AbstractGradientDescent


class AdagradGradientDescent(AbstractGradientDescent):
    """Adagrad Gradient Descent Class

    This class implements the adaptive gradient algorithm for updating the
    particles in Stein variational gradient descent. Adagrad computes individual
    gradients for each parameter by exponentially weighting the previous squared
    gradients and treating this as a normalizing factor.
    """
    def __init__(self, learning_rate=1e-3, decay=1., alpha=0.9):
        """Initialize the parameters of the Adagrad gradient descent object.

            learning_rate (float, optional): A global step size parameter used
                for every gradient descent class. This determines the global
                magnitude of a gradient descent step by scaling the gradient by
                this value. For simple models such as logistic regression, this
                value can be set relatively large (e.g. 1e-1), but must be small
                (e.g. 1e-4) for complex models such as neural networks.
            decay (float, optional): The learning rate decay parameter. After
                each gradient descent update, the learning rate is multiplied by
                this amount; this is done to guarantee convergence of the
                learning algorithm.
            alpha (float, optional): The control parameter for how much to decay
                previous squared gradients when computing the weighted average
                of the current squared gradient and the previous squared
                gradients.
        """
        super(AdagradGradientDescent, self).__init__(learning_rate, decay)
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
