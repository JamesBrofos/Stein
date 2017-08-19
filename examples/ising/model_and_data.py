import numpy as np
import tensorflow as tf
from tensorflow.contrib.distributions import Normal, Gamma
from utilities import enumerate_binary, construct_pairwise_interactions


# Load in the coronary heart disease dataset or synthetic data.
use_synthetic = True
if use_synthetic:
    path = "./data/synthetic/samples.csv"
else:
    path = "./data/applied/coronary.csv"
X = np.loadtxt(path, delimiter=",")

# Number of binary random variables.
n_samples, n = X.shape
# Number of particles to sample.
n_particles = 2000

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
