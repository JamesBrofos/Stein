import theano.tensor as T
import theano
import numpy as np
import random
import time
from functools import partial
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import train_test_split
from stein import SteinSampler
from stein.kernels import SquaredExponentialKernel
from stein.gradient_descent import Adagrad, Adam
from construct_neural_network import construct_neural_network
from utils import pack_weights, unpack_weights, normalization, init_weights


# For reproducibility.
if False:
    seed = 10
    random.seed(seed)
    np.random.seed(seed)

# Import data.
data = np.loadtxt('./data/boston_housing')
# Extract the target variable and explanatory features.
data_X = data[:, :-1]
data_y = data[:, -1]
# Number of explanatory variables.
n_vars = data_X.shape[1]

# Prior variance variables.
alpha, beta = 1.0, 0.1

# Training parameters.
n_batch = 100
n_hidden = 50
n_iters = 10000
n_particles = 20
# Total number of weights, biases, and variance parameters in the Bayesian
# neural network. Each particle will correspond to a point in this number of
# dimensions.
n_params = n_vars * n_hidden + n_hidden * 2 + 3

# Extract a Theano function for prediction and one for computing the gradient of
# the log-posterior.
prediction_func, grad_log_p_func = construct_neural_network(
    n_params, alpha, beta
)


def train(X_train, y_train):
    """Sample code to reproduce our results for the Bayesian neural network
    example. Our settings are almost the same as Hernandez-Lobato and Adams. Our
    implementation is also based on their Python code. The mathematical model is
    defined as follows:

    p(y | W, X, gamma) = \prod_i^N  N(y_i | f(x_i; W), gamma^{-1})
    p(W | lambda) = \prod_i N(w_i | 0, lambda^{-1})
    p(gamma) = Gamma(gamma | alpha, beta)
    p(lambda) = Gamma(lambda | alpha, beta)

    The posterior distribution is as follows:

    p(W, gamma, lambda) = p(y | W, X, gamma) p(W | lambda) p(gamma) p(lambda)

    To avoid negative values of gamma and lambda, we update log-gamma and
    log-lambda instead.
    """
    # Build a development set of training data. This is used to tune the
    # noise variance and produce a posterior distribution with more-likely
    # predictions.
    size_dev = min(int(np.round(0.1 * X_train.shape[0])), 500)
    X_dev, y_dev = X_train[-size_dev:], y_train[-size_dev:]
    X_train, y_train = X_train[:-size_dev], y_train[:-size_dev]

    # Normalize the training dataset and compute the number of observations in
    # the training dataset.
    X_train, y_train = normalization(
        (X_train, mean_X_train, std_X_train),
        (y_train, mean_y_train, std_y_train)
    )
    n_train = X_train.shape[0]

    # Initialize the particles for Stein variational gradient descent. The
    # weights and biases are all sampled from the prior distribution, but the
    # gamma variance is computed empirically from the mean squared error.
    theta = np.zeros([n_particles, n_params])
    for i in range(n_particles):
        w1, b1, w2, b2, loglambda = init_weights(alpha, beta, n_vars, n_hidden)
        # Empirical estimate of the noise variance.
        batch = np.random.choice(n_train, min(n_train, 1000), replace=False)
        y_hat = prediction_func(X_train[batch], w1, b1, w2, b2)
        loggamma = -np.log(np.mean((y_hat - y_train[batch])**2))
        theta[i] = pack_weights(w1, b1, w2, b2, loggamma, loglambda)

    def grad_log_p(theta):
        """Computes the gradient of a Bayesian neural network with respect to
        the weights and biases in addition to two variance parameters.

        Notice that this code implements a single layer neural network.
        """
        batch = np.random.choice(n_train, n_batch, replace=False)
        X, y = X_train[batch], y_train[batch]
        w1, b1, w2, b2, loggamma, loglambda = unpack_weights(
            theta, n_vars, n_hidden
        )
        return pack_weights(*grad_log_p_func(
            X, y, w1, b1, w2, b2, loggamma, loglambda, n_train
        ))

    # Create the kernel for Stein variational gradient descent.
    kernel = SquaredExponentialKernel(n_params)
    # Gradient descent algorithm.
    gd = Adam()
    # Perform Stein variational gradient descent.
    stein = SteinSampler(grad_log_p, kernel, gd, evaluator=evaluator)
    theta = stein.sample(
        n_particles, n_iters, learning_rate=1e-3, theta_init=theta
    )

    # Normalize the development dataset.
    X_dev = normalization((X_dev, mean_X_train, std_X_train))
    # Iterate over each particle.
    for i in range(n_particles):
        # Extract the weights, biases, and variances for this particle.
        w1, b1, w2, b2, loggamma, loglambda = unpack_weights(
            theta[i], n_vars, n_hidden
        )
        # Compute the predicted mean for a neural network with the provided
        # weights and biases. Notice that we scale the output by multiplying by
        # the standard deviation and adding the mean of the training set target
        # variable.
        pred_y_dev = (
            prediction_func(X_dev, w1, b1, w2, b2) * std_y_train + mean_y_train
        )

        def log_likelihood(loggamma):
            """Compute the log-likelihood of the data given the predictive mean
            produced by the neural network with the input specifying the
            posterior precision.
            """
            return np.sum(np.log(
                np.sqrt(np.exp(loggamma)) /np.sqrt(2*np.pi) *
                np.exp(-(pred_y_dev - y_dev)**2 / 2 * np.exp(loggamma))
            ))

        # Select the noise precision using a development set. This allows us to
        # make more likely predictions under the posterior.
        lik1 = log_likelihood(loggamma)
        # Compute the empirical noise precision.
        loggamma = -np.log(np.mean(np.power(pred_y_dev - y_dev, 2)))
        lik2 = log_likelihood(loggamma)
        # If using the empirical precision yields a better fit to the
        # development dataset, then update the log-gamma parameter.
        if lik2 > lik1:
            theta[i,-2] = loggamma

    return theta


def evaluation(theta, X_test, y_test):
    """Evaluate the performance of the Bayesian neural network by computing its
    root mean squared error and log-likelihood on a test dataset. Notice that
    this function performs normalization based on the mean and variance
    parameters of the training data.
    """
    # Normalize the test dataset.
    X_test = normalization((X_test, mean_X_train, std_X_train))

    # Construct vectors to store the prediction for each of the particles and
    # each of the test data points, as well as the likelihood of the prediction
    # under the posterior.
    pred_y_test = np.zeros([n_particles, len(y_test)])
    prob = np.zeros([n_particles, len(y_test)])

    # We adopt a Bayesian perspective on computing the metrics by averaging over
    # the predictions of each constituent particle.
    for i in range(n_particles):
        # Extract the weights, biases, and variance parameters for this
        # particle.
        w1, b1, w2, b2, loggamma, loglambda = unpack_weights(
            theta[i], n_vars, n_hidden
        )
        # Compute the test set predictions.
        pred_y_test[i] = (
            prediction_func(X_test, w1, b1, w2, b2) * std_y_train + mean_y_train
        )
        # Compute the test set log-likelihood under the posterior.
        prob[i] = (
            np.sqrt(np.exp(loggamma)) / np.sqrt(2*np.pi) *
            np.exp(-(pred_y_test[i] - y_test)**2 / 2 * np.exp(loggamma) )
        )
    # Average predictions across particles.
    pred = np.mean(pred_y_test, axis=0)
    # Evaluation.
    rmse = np.sqrt(np.mean((pred - y_test)**2))
    ll = np.mean(np.log(np.mean(prob, axis = 0)))

    return rmse, ll


# Partition the dataset into a training and testing components, with 90% and 10%
# of the total data, respectively.
train_ratio = 0.9
permutation = np.arange(data_X.shape[0])
random.shuffle(permutation)
size_train = int(np.round(data_X.shape[0] * train_ratio))
index_train = permutation[0:size_train]
index_test = permutation[size_train :]
# Partition based on random assignment.
X_train, y_train = data_X[index_train, :], data_y[index_train]
X_test, y_test = data_X[index_test, :], data_y[index_test]
# Create a function to serve as an evaluator in the Stein sampler class.
evaluator = partial(evaluation, X_test=X_test, y_test=y_test)

# Compute standardization variables to scale the inputs to have mean zero and
# unit variance.
mean_X_train, std_X_train = np.mean(X_train, 0), np.std(X_train, 0)
mean_y_train, std_y_train = np.mean(y_train), np.std(y_train)
std_X_train[std_X_train == 0] = 1


# Keep track of time.
start = time.time()
# Estimate the parameters of a Bayesian neural network using Stein variational
# gradient descent.
theta = train(X_train, y_train)
# Compute performance metrics.
time_elapsed = time.time() - start
rmse, ll = evaluation(theta, X_test, y_test)

# Show performance diagnostics.
print("Root mean squared error:\t{}".format(rmse))
print("Posterior log-likelihood:\t{}".format(ll))
print("Time elapsed:\t\t\t\t\t\t\t{}".format(time_elapsed))
