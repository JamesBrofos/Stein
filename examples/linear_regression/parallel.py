import numpy as np
from time import time
from stein.gradient_descent import AdamGradientDescent
from stein.samplers import ParallelSteinSampler
from model_and_data import log_p, model_X, model_y, data_X, data_y, data_w


# Record time elapsed.
start_time = time()
# Number of learning iterations.
n_iters = 500
# Sample from the posterior using Stein variational gradient descent.
n_particles = 80
n_shuffle = 1
gd = AdamGradientDescent(learning_rate=1e-1)
sampler = ParallelSteinSampler(n_particles, n_shuffle, log_p, gd)
# Perform learning iterations.
for i in range(n_iters):
    sampler.train_on_batch({model_X: data_X, model_y: data_y})

# Combine all of the particles.
theta = sampler.merge()
# Show diagnostics.
if sampler.is_master:
    vals = list(theta.values())[0]
    est = np.array(vals.mean(axis=0)).ravel()
    print("True coefficients: {}".format(data_w.ravel()))
    print("Est. coefficients: {}".format(est))
    print("Time elapsed: {}".format(time() - start_time))
