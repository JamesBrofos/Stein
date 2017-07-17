import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.contrib.distributions import Normal, Gamma
from stein import SteinSampler
from stein.gradient_descent import AdamGradientDescent


# For reproducibility.
np.random.seed(0)

# Generate random data from a logistic regression model.
n_samples, n_feats = 1000, 1
data_X = np.random.normal(size=(n_samples, n_feats))
data_w = np.random.normal(scale=3., size=(n_feats, 1))
data_y = np.random.normal(data_X.dot(data_w), 0.1)

with tf.variable_scope("model"):
    # Placeholders for features and targets.
    model_X = tf.placeholder(tf.float32, shape=[None, n_feats])
    model_y = tf.placeholder(tf.float32, shape=[None, 1])
    model_w = tf.Variable(tf.zeros([n_feats, 1]))
    # Compute prior.
    with tf.variable_scope("priors"):
        w_prior = Normal(tf.zeros([n_feats, 1]), 1.)
    # Compute likelihood function.
    with tf.variable_scope("likelihood"):
        y_hat = tf.matmul(model_X, model_w)
        log_l = -0.5 * tf.reduce_sum(tf.square(y_hat - model_y))
    # Compute the log-posterior of the model.
    log_p = log_l + tf.reduce_sum(w_prior.log_prob(model_w))


# Number of learning iterations.
n_iters = 1000
n_prog = n_iters // 10
# Sample from the posterior using Stein variational gradient descent.
n_particles = 10
gd = AdamGradientDescent(learning_rate=1e-1)
sampler = SteinSampler(n_particles, log_p, gd)
# Perform learning iterations.
for i in range(n_iters):
    if i % n_prog == 0:
        print("Iteration: {} / {}".format(i, n_iters))
    sampler.train_on_batch({model_X: data_X, model_y: data_y})

# Visualize if there is only a single dimension.
if n_feats == 1:
    X_plot = np.atleast_2d(np.linspace(-3., 3., num=100)).T
    plt.plot(data_X.ravel(), data_y.ravel(), "r.")
    for i in range(n_particles):
        y_plot = X_plot.dot(sampler.theta[model_w][i])
        plt.plot(X_plot.ravel(), y_plot.ravel(), "g-", alpha=0.5)
    y = X_plot.dot(data_w)
    plt.plot(X_plot.ravel(), y.ravel(), "b-", linewidth=2.)
    plt.grid()
    plt.show()
