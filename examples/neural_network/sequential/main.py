import numpy as np
import tensorflow as tf
import random
from time import time
from tensorflow.contrib.distributions import Normal, Gamma
from sklearn.model_selection import train_test_split
from stein.samplers import SteinSampler
from stein.gradient_descent import AdamGradientDescent


# For reproducibility.
if True:
    np.random.seed(1)

# Import data.
dataset = "../data/boston_housing.txt"
data = np.loadtxt(dataset)
# Extract the target variable and explanatory features.
data_X = data[:, :-1]
data_y = data[:, -1:]

# Partition the dataset into a training and testing components, with 90% and 10%
# of the total data, respectively.
train_ratio = 0.9
permutation = np.random.permutation(data_X.shape[0])
size_train = int(np.round(data_X.shape[0] * train_ratio))
index_train = permutation[:size_train]
index_test = permutation[size_train:]
# Partition based on random assignment.
X_train, y_train = data_X[index_train, :], data_y[index_train]
X_test, y_test = data_X[index_test, :], data_y[index_test]

# Normalize the data. Notice that we do not normalize the test targets as this
# would result in non-comparable root mean squared errors.
X_train_mean, X_train_std = np.mean(X_train, 0), np.std(X_train, 0)
y_train_mean, y_train_std = np.mean(y_train), np.std(y_train)
X_train = (X_train - X_train_mean) / X_train_std
X_test = (X_test - X_train_mean) / X_train_std
y_train = (y_train - y_train_mean) / y_train_std
# Number of explanatory variables.
n_train, n_feats = X_train.shape

# Parameters for training such as the number of hidden neurons and the batch
# size to use during training, the total number of training iterations, and the
# number of particles to sample from the posterior.
n_hidden = 50
n_batch = 100
n_prog = 1
n_particles = 30
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
        log_l = Normal(pred, tf.reciprocal(tf.sqrt(model_gamma)))
    # Priors.
    with tf.variable_scope("priors"):
        prior_gamma = Gamma(alpha, beta)
        prior_lambda = Gamma(alpha, beta)
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
        tf.reduce_sum(log_l.log_prob(model_y)) * n_train / n_batch +
        # Variance priors.
        prior_gamma.log_prob(model_gamma) +
        prior_lambda.log_prob(model_lambda) +
        # Weight and bias priors.
        tf.reduce_sum(prior_w_1.log_prob(model_w_1)) +
        tf.reduce_sum(prior_w_2.log_prob(model_w_2)) +
        tf.reduce_sum(prior_b_1.log_prob(model_b_1)) +
        prior_b_2.log_prob(model_b_2)
    ) / n_train

# Initialize the dictionary of parameters.
theta = {
    model_w_1: (
        1. / np.sqrt(n_feats + n_hidden) *
        np.random.randn(n_particles, n_feats, n_hidden)
    ),
    model_w_2: (
        1. / np.sqrt(n_hidden + 1) * np.random.randn(n_particles, n_hidden, 1)
    ),
    model_b_1: np.zeros((n_particles, n_hidden)),
    model_b_2: np.zeros((n_particles, )),
    model_log_lambda: np.log(
        np.random.gamma(alpha, beta, size=(n_particles, ))
    ),
    model_log_gamma: np.zeros((n_particles, ))
}
# This is a better initialization of the noise variance parameter.
with tf.Session() as sess:
    data_feed = {model_X: X_train, model_y: y_train}
    for i in range(n_particles):
        current_theta = {v: x[i] for v, x in theta.items()}
        current_theta.update(data_feed)
        y_hat = sess.run(pred, current_theta).ravel()
        theta[model_log_gamma][i] = -np.log(
            np.mean((y_train.ravel() - y_hat)**2)
        )

def evaluate(sampler, data_feed):
    """Evaluate the performance of the Bayesian neural network by computing its
    root mean squared error and log-likelihood on a test dataset. Notice that
    this function performs normalization based on the mean and variance
    parameters of the training data.
    """
    # Construct vectors to store the prediction for each of the particles and
    # each of the test data points under the posterior.
    y_test_pred = np.zeros([n_particles, data_feed[model_y].shape[0]])
    # We adopt a Bayesian perspective on computing the metrics by averaging over
    # the predictions of each constituent particle.
    for i in range(n_particles):
        feed = {v: x[i] for v, x in sampler.theta.items()}
        feed.update(data_feed)
        y_test_pred[i] = (
            sampler.sess.run(pred, feed) * y_train_std + y_train_mean
        ).ravel()

    # Average predictions across particles.
    avg_pred = np.mean(y_test_pred, axis=0)
    # Evaluation.
    rmse = np.sqrt(np.mean((avg_pred - data_feed[model_y].ravel())**2))

    return rmse

# Gradient descent object.
gd = AdamGradientDescent(learning_rate=1e-1)
sampler = SteinSampler(n_particles, log_p, gd, theta)
# Perform Stein variational gradient descent to sample from the posterior
# distribution of the Bayesian neural network.
current_iter = 0
start_time = time()
while True:
    # Increment the global number of learning iterations.
    current_iter += 1
    # Train on batch.
    batch = np.random.choice(n_train, n_batch, replace=False)
    X, y = X_train[batch], y_train[batch]
    sampler.train_on_batch({model_X: X, model_y: y})
    # Output diagnostic variables.
    if current_iter % n_prog == 0:
        elapsed_time = time() - start_time
        rmse_train = evaluate(sampler, {
            model_X: X_train,
            model_y: y_train * y_train_std + y_train_mean
        })
        rmse_test = evaluate(sampler, {model_X: X_test, model_y: y_test})
        print("Iteration {}:\t\t{:.4f}\t\t{:.4f}\t\t{:.6f}".format(
            current_iter, rmse_train, rmse_test, elapsed_time
        ))
        start_time = time()


