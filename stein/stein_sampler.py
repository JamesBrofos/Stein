import numpy as np


class SteinSampler(object):
    """Stein Sampler Class"""
    def __init__(
            self,
            grad_log_p,
            kernel,
            gd,
            evaluator=None,
            verbose=True
    ):
        """Initialize parameters of the Stein sampler."""
        self.grad_log_p = grad_log_p
        self.kernel = kernel
        self.gd = gd
        self.evaluator = evaluator
        self.verbose = verbose

    def __phi(self, theta):
        """Assuming a reproducing kernel Hilbert space with associated kernel,
        this function computes the optimal perturbation in the particles under
        functions in the unit ball under the norm of the RKHS. This
        perturbation can be regarded as the direction that will maximally
        decrease the KL-divergence between the empirical distribution of the
        particles and the target distribution.
        """
        # Extract the number of particles and number of parameters.
        n_particles, n_params = theta.shape
        # Compute the kernel matrices and gradient with respect to the
        # particles.
        K, dK = self.kernel.kernel_and_grad(theta)
        # Compute the gradient of the logarithm of the target density.
        g = np.zeros((n_particles, n_params))
        for i in range(n_particles):
            g[i] = self.grad_log_p(theta[i])

        return (K.dot(g) + dK) / n_particles

    def sample(
            self,
            n_particles,
            n_iters,
            theta_init=None
    ):
        """Use Stein variational gradient descent to sample from the target
        distribution by iteratively perturbing a specified number of points so
        that they are maximally close to the target.
        """
        # Number of iterations after which to provide an update, if required.
        n_prog = n_iters // 10
        # Randomly initialize a set of theta which are the particles to
        # transform so that they resemble a random sample from the target
        # distribution.
        if theta_init is not None:
            theta = theta_init
        else:
            theta = np.random.normal(size=(n_particles, self.kernel.n_params))

        # Perform Stein variational gradient descent.
        for i in range(n_iters):
            # Compute the optimal perturbation.
            theta += self.gd.update(self.__phi(theta))

            # Print out diagnostics.
            if i % n_prog == 0 and self.verbose:
                print("Iteration:\t{} / {}".format(i, n_iters))
            if i % n_prog == 0 and self.evaluator is not None:
                print("Metric:\t{}".format(self.evaluator(theta)))

        return theta
