import numpy as np
import matplotlib.pyplot as plt
from time import time
from scipy.stats import norm
from stein.samplers import SteinSampler
from stein.optimizers import AdamGradientDescent
from model_and_data import log_p, data_X, n_particles, theta


# Number of learning iterations.
n_iters = 600
# Sample from the posterior using Stein variational gradient descent.
gd = AdamGradientDescent(learning_rate=1e0)
sampler = SteinSampler(n_particles, log_p, gd, theta)
# Perform learning iterations.
for i in range(n_iters):
    if i % 100 == 0:
        print("Iteration: {}".format(i))
    sampler.train_on_batch({})

# Extract samples from Stein variational gradient descent.
theta = list(sampler.theta.values())[0]


if True:
    r = np.linspace(-4., 4., num=100)
    dens = 1./3 * norm.pdf(r, loc=-2.) + 2./3 * norm.pdf(r, loc=2.)
    plt.figure(figsize=(8, 6))
    plt.hist(theta, bins=20, normed=True)
    plt.plot(r, dens)
    plt.grid()
    plt.show()

# Show diagnostics.
w = np.random.normal()
b = np.random.uniform(0., 2.*np.pi)
print("MSE(E[X]) = {}".format(np.log(
    np.mean((data_X.mean() - theta.mean())**2))
))
print("MSE(E[X^2]) = {}".format(np.log(
    np.mean(((data_X**2).mean() - (theta**2).mean())**2))
))
print("MSE(E[cos(wX + b)]) = {}".format(np.log(
    np.mean(((np.cos(w*data_X + b)).mean() - (np.cos(w*theta + b)).mean())**2))
))
