import numpy as np
import tensorflow as tf
from time import time
from tensorflow.contrib.distributions import Normal
from stein.samplers import DistributedSteinSampler
from stein.optimizers import AdamGradientDescent


# Import data.
data_X = np.loadtxt("./data/data_X.csv", delimiter=",")
if len(data_X.shape) == 1:
    data_X = np.atleast_2d(data_X).T
data_w = np.atleast_2d(np.loadtxt("./data/data_w.csv", delimiter=",")).T
data_y = np.atleast_2d(np.loadtxt("./data/data_y.csv", delimiter=",")).T
n_samples, n_feats = data_X.shape

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

# Record time elapsed.
start_time = time()
# Number of learning iterations.
n_iters = 500
# Sample from the posterior using Stein variational gradient descent.
n_threads = 4
n_particles = 4000
gd = AdamGradientDescent(learning_rate=1e-1)
sampler = DistributedSteinSampler(n_threads, n_particles, log_p, gd)
# Perform learning iterations.
for i in range(n_iters):
    start_iter = time()
    sampler.train_on_batch({model_X: data_X, model_y: data_y})
    end_iter = time()
    print("Iteration {}. Time to complete iteration: {:.4f}".format(
        i, end_iter - start_iter
    ))

# Show diagnostics.
est = np.array(list(sampler.theta.values()))[0].mean(axis=0).ravel()
print("True coefficients: {}".format(data_w.ravel()))
print("Est. coefficients: {}".format(est))
print("Time elapsed: {}".format(time() - start_time))
