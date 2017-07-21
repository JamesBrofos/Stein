import numpy as np
import os
from time import time
from stein.samplers import SteinSampler
from stein.gradient_descent import AdamGradientDescent
from model_and_data import log_p, model_X, model_y, data_X, data_y, data_w


# Limit TensorFlow output.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Record time elapsed.
start_time = time()
# Number of learning iterations.
n_iters = 500
# Sample from the posterior using Stein variational gradient descent.
n_particles = 80
gd = AdamGradientDescent(learning_rate=1e-1)
sampler = SteinSampler(n_particles, log_p, gd)
# Perform learning iterations.
for i in range(n_iters):
    sampler.train_on_batch({
        model_X: data_X,
        model_y: data_y
    })

# Show diagnostics.
est = np.array(list(sampler.theta.values()))[0].mean(axis=0).ravel()
print("True coefficients: {}".format(data_w.ravel()))
print("Est. coefficients: {}".format(est))
print("Time elapsed: {}".format(time() - start_time))
