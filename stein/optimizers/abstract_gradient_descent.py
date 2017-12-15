from abc import ABCMeta, abstractmethod


class AbstractGradientDescent(object):
    """Abstract Gradient Descent Class

    A gradient descent object is characterized by both its global learning rate
    as well as its update scheme. This class has an abstract method
    corresponding to the update scheme (such as Adagrad or Adam), which is
    implemented by classes inheriting from the abstract gradient descent class.
    The global learning rate is set in the initialization method.
    """
    def __init__(self, learning_rate, decay):
        """Initialize the parameters of the abstract gradient descent object.

        Parameters:
            learning_rate (float): A global step size parameter used for every
                gradient descent class. This determines the global magnitude of
                a gradient descent step by scaling the gradient by this value.
                For simple models such as logistic regression, this value can be
                set relatively large (e.g. 1e-1), but must be small (e.g. 1e-4)
                for complex models such as neural networks.
            decay (float): The learning rate decay parameter. After each
                gradient descent update, the learning rate is multiplied by this
                amount; this is done to guarantee convergence of the learning
                algorithm.
        """
        self.learning_rate = learning_rate
        self.decay = decay
        self.n_iters = 0

    @abstractmethod
    def update(self, phi):
        """Compute the update direction for the gradient descent algorithm.

        Parameters:
            phi (numpy array): A numpy array corresponding to the optimal
                perturbation computed using the Stein variational gradient
                descent algorithm. The dimensions of this matrix are the number
                of particles by the number of model parameters. This value will
                be augmented according to the update scheme of the gradient
                descent algorithm and then applied to the current model
                particles (in matrix representation).

        Returns:
            Numpy array: A numpy array corresponding to the augmented optimal
                perturbation direction computed using Stein variational gradient
                descent. This value is added to the current values of the model
                parameters, which an individual update direction for every
                parameter of every particle.
        """
        raise NotImplementedError()

