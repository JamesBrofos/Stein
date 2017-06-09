import numpy as np


class SteinSampler(object):
    """Stein Sampler Class"""
    def __init__(self, grad_log_p, kernel, verbose=True):
        """Initialize parameters of the Stein sampler."""
        self.grad_log_p = grad_log_p
        self.kernel = kernel
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

        # return (np.matmul(K, g) + dK) / n_particles
        return (K.dot(g) + dK) / n_particles

    def sample(
            self,
            n_particles,
            n_iters,
            learning_rate=1e-3,
            alpha=0.9,
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
            if i % n_prog == 0 and self.verbose:
                print("Iteration:\t{} / {}".format(i, n_iters))
            # Compute the optimal perturbation.
            perturb = self.__phi(theta)
            # Use Adagrad to update the particles.
            if i == 0:
                hist = perturb ** 2
            else:
                hist = alpha * hist + (1. - alpha) * (perturb ** 2)

            grad = perturb / (1e-6 + np.sqrt(hist))
            theta += learning_rate * grad

        return theta
