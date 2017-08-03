import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.stats import norm
from stein.samplers import SteinSampler
from stein.gradient_descent import AdamGradientDescent
from model_and_data import (
    log_p,
    log_l,
    X,
    XP,
    model_W,
    model_b,
    model_X_W,
    model_X_b,
    model_log_alpha
)


# Number of learning iterations.
n_iters = 1000
n_prog = 1
# Sample from the posterior using Stein variational gradient descent.
n_particles = 100
gd = AdamGradientDescent(learning_rate=1e-1, decay=0.9999)
sampler = SteinSampler(n_particles, log_p, gd)

# Get the real weights and biases.
W = np.loadtxt("./data/weights.csv", delimiter=",")
b = np.loadtxt("./data/biases.csv", delimiter=",")
# Define a function to measure the mean squared error between estimated weights
# and biases and the true parameters.
weights_mse = 2. * tf.nn.l2_loss(W - tf.squeeze(model_W))
biases_mse = 2. * tf.nn.l2_loss(b - tf.squeeze(model_b))

# Perform learning iterations.
for i in range(n_iters):
    # Train on batch.
    batch_feed = {model_X_W: XP, model_X_b: X}
    sampler.train_on_batch(batch_feed)
    if i % n_prog == 0:
        log_likelihood = np.mean(sampler.function_posterior_distribution(
            log_l, batch_feed
        ))
        W_mse = np.mean(sampler.function_posterior_distribution(
            weights_mse, batch_feed
        ))
        b_mse = np.mean(sampler.function_posterior_distribution(
            biases_mse, batch_feed
        ))
        print("Iteration {} / {}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}".format(
            i, n_iters, log_likelihood, W_mse, b_mse
        ))




