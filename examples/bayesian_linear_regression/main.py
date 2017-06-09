import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from stein.kernels import SquaredExponentialKernel
from stein import SteinSampler

# For reproducibility.
np.random.seed(0)

# Number of observations and number of covariates.
n = 100
k = 5
# Create true linear coefficients and true noise variance.
beta = np.ones((k, ))
sigma_sq = 1.

# Generate synthetic data from a linear model.
X = np.random.normal(size=(n, k))
y = np.random.normal(X.dot(beta), sigma_sq)


def grad_log_p(theta):
    """Gradient of the log-posterior distribution with respect to the linear
    coefficients in a Bayesian linear regression model.
    """
    e = y - X.dot(theta)
    g = e.dot(X) - theta
    return g

# Setup parameters of the Stein sampler.
n_particles = 100
n_params = k
n_iters = 1000
# Specify that the Stein sampler should use a squared exponential kernel.
kernel = SquaredExponentialKernel(n_params)

# Create the Stein sampler.
stein = SteinSampler(grad_log_p, kernel)
# Sample using Stein variational gradient descent with a squared exponential
# kernel on the posterior distribution over linear coefficients in a Bayesian
# linear model.
theta = stein.sample(n_particles, n_iters, learning_rate=1e-2)

# Now compare to the closed-form solution by computing the posterior mean and
# posterior covariance.
posterior_cov = np.linalg.inv(X.T.dot(X) + np.eye(k))
posterior_mean = posterior_cov.dot(X.T.dot(y))
empirical_mean = theta.mean(axis=0)
empirical_cov = np.atleast_2d(np.cov(theta.T))

# Show diagnostics.
print("Theoretical mean:\t{}".format(posterior_mean))
print("Empirical mean:\t\t{}".format(empirical_mean))
print("Theoretical covariance:")
print(posterior_cov)
print("Empirical covariance:")
print(empirical_cov)

