import numpy as np
import tensorflow as tf
from tensorflow.contrib.distributions import Normal, Gamma
from stein.samplers import SteinSampler
from stein.gradient_descent import AdamGradientDescent
from utilities import enumerate_binary, construct_pairwise_interactions


# Load in the coronary heart disease dataset.
X = np.loadtxt("./data/coronary.csv", delimiter=",")
# Number of binary random variables.
n_samples, n = X.shape

# Load in the binarization of the space.
E = enumerate_binary(n)
# Make pairwise interactions.
XP = construct_pairwise_interactions(X)
EP = construct_pairwise_interactions(E)

# Number of weights. The number of biases in the Boltzmann Machine is simply the
# number of binary random variables.
p = (n * (n - 1)) // 2

# Define the Boltzmann Machine probabilistic model. This can also be thought of
# as a binarized Ising model.
with tf.variable_scope("model"):
    # Define placeholders for both the pairwise interactions and the biases.
    model_X_b = tf.placeholder(tf.float32, shape=[None, n])
    model_X_W = tf.placeholder(tf.float32, shape=[None, p])
    # Define variables for the weights and biases of the Boltzmann Machine.
    model_W = tf.Variable(tf.zeros([p, 1]), dtype=tf.float32)
    model_b = tf.Variable(tf.zeros([n, 1]), dtype=tf.float32)
    model_log_alpha = tf.Variable(tf.zeros([]))
    model_alpha = tf.exp(model_log_alpha)

    # Place priors over the weights and biases.
    with tf.variable_scope("priors"):
        W_prior = Normal(
            tf.zeros([p, 1]),
            tf.reciprocal(tf.sqrt(model_alpha))
        )
        b_prior = Normal(
            tf.zeros([n, 1]),
            tf.reciprocal(tf.sqrt(model_alpha))
        )
        alpha_prior = Gamma(1., 0.01)
    # Compute the likelihood function.
    with tf.variable_scope("likelihood"):
        log_l = (
            # Observations.
            tf.reduce_mean(
                tf.matmul(model_X_W, model_W) + tf.matmul(model_X_b, model_b)
            ) -
            # Partition function.
            tf.reduce_logsumexp(
                tf.matmul(EP, model_W) + tf.matmul(E, model_b)
            )
        )
    # Compute the log-posterior of the model.
    log_p = (
        log_l * n_samples +
        tf.reduce_sum(W_prior.log_prob(model_W)) +
        tf.reduce_sum(b_prior.log_prob(model_b)) +
        alpha_prior.log_prob(model_alpha)
    )

# Number of learning iterations.
n_iters = 10000
n_prog = 100
# Sample from the posterior using Stein variational gradient descent.
n_particles = 100
gd = AdamGradientDescent(learning_rate=1e-1, decay=0.9999)
sampler = SteinSampler(n_particles, log_p, gd)
# Perform learning iterations.
for i in range(n_iters):
    # Train on batch.
    batch_feed = {model_X_W: XP, model_X_b: X}
    sampler.train_on_batch(batch_feed)
    if i % n_prog == 0:
        l = np.zeros((n_particles, ))
        for j in range(n_particles):
            batch_feed.update({v: x[j] for v, x in sampler.theta.items()})
            l[j] = sampler.sess.run(log_l, batch_feed)
        log_likelihood = np.mean(l)
        print("Iteration {} / {}\t\t{:.4f}".format(i, n_iters, log_likelihood))

