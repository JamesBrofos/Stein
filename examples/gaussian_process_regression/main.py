import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from stein.kernels import SquaredExponentialKernel
from stein import SteinSampler


n = 50
X = np.random.uniform(low=-1., high=1., size=(n, 1))
y = np.random.normal(np.sin(5.*X), 0.1).ravel()


# Visualize the training data.
if False:
    plt.figure(figsize=(8, 6))
    plt.plot(X.ravel(), y, ".")
    plt.grid()
    plt.show()


def gaussian_kernel(theta, X, Y=None):
    """Compute the covariance between inputs according to the squared
    exponential kernel.
    """
    # Compute the squared distance between inputs.
    Xp = X / theta
    r_sq = cdist(Xp, Xp, "sqeuclidean")
    # Return the Gaussian kernel for the given inputs and length scales.
    return np.exp(-0.5 * r_sq)

def grad_gaussian_kernel(theta):
    K = gaussian_kernel(theta, X)
    return (
        (np.expand_dims(X, axis=1) - np.expand_dims(X, axis=0)) ** 2 /
        (theta ** 3)
    ) * K[..., np.newaxis]

def grad_log_p(theta):
    """Gradient of the Gaussian process likelihood with respect to the length
    scales.
    """
    K = gaussian_kernel(theta, X)
    K_inv = np.linalg.inv(K)
    alpha = K_inv.dot(y)
    O = np.outer(alpha, alpha)
    g = grad_gaussian_kernel(theta)
    return 0.5 * np.trace((O - K_inv).dot(g)) - 0.5 * 0.1 * theta.dot(theta)

# Setup parameters of the Stein sampler.
n_particles = 100
n_params = 1
n_iters = 1000
# Specify that the Stein sampler should use a squared exponential kernel.
kernel = SquaredExponentialKernel(n_params)

# Create the Stein sampler.
stein = SteinSampler(grad_log_p, kernel)
# Sample using Stein variational gradient descent with a squared exponential
# kernel on the posterior distribution over the parameters of a Bayesian
# logistic regression model.
theta = stein.sample(n_particles, n_iters, learning_rate=1e-2)

if True:
    plt.figure(figsize=(8, 6))
    plt.hist(theta, bins=20, normed=True)
    plt.grid()
    plt.show()


# Evaluate the quality of the model.
# X_test = np.atleast_2d(np.linspace(-1., 1., num=100)).T

# K_pred = gaussian_kernel()
