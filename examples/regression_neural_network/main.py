import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import io
from sklearn.model_selection import train_test_split
from tensorflow.contrib.distributions import Normal, Gamma
from stein.samplers import SteinSampler
from stein.optimizers import AdamGradientDescent


# Generate synthetic data.
X_train = np.random.uniform(size=(20, 1))
y_train = np.random.normal(np.cos(10 * X_train) * (5 * X_train), 0.1)

# Parameters for training such as the number of hidden neurons and the batch
# size to use during training, the total number of training iterations, and the
# number of particles to sample from the posterior.
n_hidden = 100
n_batch = 20
n_particles = 20
# Number of training data points.
n_train, n_feats = X_train.shape
# Extract learning rate.
learning_rate = 1e-1
# Precision prior parameters.
alpha, beta = 1., 0.01

# Define a Bayesian neural network model in TensorFlow.
with tf.variable_scope("model"):
    # Placeholder variables for data.
    model_X = tf.placeholder(tf.float32, shape=[None, n_feats])
    model_y = tf.placeholder(tf.float32, shape=[None, 1])
    # TensorFlow variables for the networks weights and biases, the precision
    # over the weights and biases, and the noise precision of the output.
    model_log_lambda = tf.Variable(tf.zeros([]))
    model_log_gamma = tf.Variable(tf.zeros([]))
    model_lambda = tf.exp(model_log_lambda)
    model_gamma = tf.exp(model_log_gamma)
    model_w_1 = tf.Variable(tf.zeros([n_feats, n_hidden]))
    model_b_1 = tf.Variable(tf.zeros([n_hidden]))
    model_w_2 = tf.Variable(tf.zeros([n_hidden, 1]))
    model_b_2 = tf.Variable(tf.zeros([]))

    # Compute the prediction from the network.
    with tf.variable_scope("prediction"):
        pred = tf.matmul(
            tf.nn.relu(tf.matmul(model_X, model_w_1) + model_b_1), model_w_2
        ) + model_b_2
    # Likelihood function.
    with tf.variable_scope("likelihood"):
        log_l_dist = Normal(pred, tf.reciprocal(tf.sqrt(model_gamma)))
        log_l = tf.reduce_sum(log_l_dist.log_prob(model_y))
    # Priors.
    with tf.variable_scope("priors"):
        prior_lambda = Gamma(alpha, beta)
        prior_gamma = Gamma(alpha, beta)
        prior_w_1 = Normal(
            tf.zeros([n_feats, n_hidden]),
            tf.reciprocal(tf.sqrt(model_lambda))
        )
        prior_b_1 = Normal(
            tf.zeros([n_hidden]),
            tf.reciprocal(tf.sqrt(model_lambda))
        )
        prior_w_2 = Normal(
            tf.zeros([n_hidden, 1]),
            tf.reciprocal(tf.sqrt(model_lambda))
        )
        prior_b_2 = Normal(
            tf.zeros([]),
            tf.reciprocal(tf.sqrt(model_lambda))
        )
    # Compute the log-posterior distribution.
    log_p = (
        # Rescaled log-likelihood (to account for batch updates).
        log_l * n_train / n_batch +
        # Variance priors.
        prior_lambda.log_prob(model_lambda) +
        prior_gamma.log_prob(model_gamma) +
        # Weight and bias priors.
        tf.reduce_sum(prior_w_1.log_prob(model_w_1)) +
        tf.reduce_sum(prior_w_2.log_prob(model_w_2)) +
        tf.reduce_sum(prior_b_1.log_prob(model_b_1)) +
        prior_b_2.log_prob(model_b_2)
    ) / n_train

# Gradient descent object.
gd = AdamGradientDescent(learning_rate=learning_rate, decay=0.999)
# Perform Stein variational gradient descent to sample from the posterior
# distribution of the Bayesian neural network.
sampler = SteinSampler(n_particles, log_p, gd)
# Number of iterations to perform before displaying diagnostics.
n_prog = 1000
# Perform learning iterations.
for i in range(10000):
    batch = np.random.choice(n_train, n_batch, replace=False)
    X, y = X_train[batch], y_train[batch]
    sampler.train_on_batch({model_X: X, model_y: y})
    if i % n_prog == 0:
        y_hat = sampler.function_posterior(pred, {model_X: X_train})
        mse = np.mean((y_train.ravel() - y_hat.mean(axis=0)) ** 2)
        print("Iteration: {}. Mean squared error: {:.4f}".format(i, mse))

if True:
    r = np.atleast_2d(np.linspace(0., 1.5, num=200)).T
    y_vis = sampler.function_posterior(pred, {model_X: r})
    plt.figure()
    plt.plot(X_train.ravel(), y_train.ravel(), "r.")
    for i in range(n_particles):
        plt.plot(r.ravel(), y_vis[i], "b-", alpha=0.3)
    plt.grid()
    plt.show()
