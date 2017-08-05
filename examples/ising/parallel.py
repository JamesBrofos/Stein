import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.stats import norm
from stein.samplers import ParallelSteinSampler
from stein.gradient_descent import AdamGradientDescent
from model_and_data import (
    use_synthetic,
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
n_shuffle = 10
n_prog = 1
# Sample from the posterior using Stein variational gradient descent.
n_particles = 1000
gd = AdamGradientDescent(learning_rate=1e-1, decay=0.9999)
sampler = ParallelSteinSampler(n_particles, n_shuffle, log_p, gd)

if use_synthetic:
    # Get the real weights and biases.
    W = np.loadtxt("./data/synthetic/weights.csv", delimiter=",")
    b = np.loadtxt("./data/synthetic/biases.csv", delimiter=",")
    # Define a function to measure the mean squared error between estimated
    # weights and biases and the true parameters.
    weights_mse = 2. * tf.nn.l2_loss(W - tf.squeeze(model_W))
    biases_mse = 2. * tf.nn.l2_loss(b - tf.squeeze(model_b))

# Perform learning iterations.
for i in range(n_iters):
    # Train on batch.
    batch_feed = {model_X_W: XP, model_X_b: X}
    sampler.train_on_batch(batch_feed)
    if i % n_prog == 0:
        log_likelihood = sampler.function_posterior(log_l, batch_feed, axis=0)
        if use_synthetic:
            W_mse = sampler.function_posterior(weights_mse, batch_feed, axis=0)
            b_mse = sampler.function_posterior(biases_mse, batch_feed, axis=0)
            if sampler.is_master:
                print("Iteration {} / {}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}".format(
                    i, n_iters, log_likelihood[0], W_mse[0], b_mse[0]
                ))
        else:
            if sampler.is_master:
                print("Iteration {} / {}\t\t{:.4f}".format(
                    i, n_iters, log_likelihood
                ))
