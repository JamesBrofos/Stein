import numpy as np
from functools import partial
from .stein_sampler import compute_phi


class SteinAmortizedMLE(object):
    """Stein Amortized Maximum Likelihood Estimator (MLE) Class"""
    def __init__(self, grad_log_p, grad_log_f, kernel, generator):
        """Initialize the parameters of the Stein amortized MLE object."""
        self.grad_log_p = grad_log_p
        self.grad_log_f = grad_log_f
        self.kernel = kernel
        self.generator = generator

    def amortize(self, X, n_particles, n_iters, n_latent, n_batch):
        """Alternate between estimating the parameters of the probabilistic
        model and the generative model.
        """
        # Extract the number of training examples.
        n_train = X.shape[0]

        # Iteratively perform amortized maximum likelihood estimation using
        # Stein variational gradient descent to update the parameters of the
        # generator.
        for i in range(n_iters):
            # Sample from the prior noise distribution.
            epsilon = np.random.uniform(-1., 1., size=(n_particles, n_latent))
            # Sample minibatch of data that are truly drawn from the
            # distribution.
            batch = np.random.choose(n_train, n_batch)
            X_pos = X[batch]
            X_neg = generator(epsilon)

            # Compute the optimal perturbation direction using the maximum
            # violation of Stein's identity.
            grad_log_p_params_fixed = partial(self.grad_log_p, params=theta)
            phi = compute_phi(X_neg, self.kernel, grad_log_p_params_fixed)
            
            
