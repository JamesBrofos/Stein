import numpy as np
from functools import partial
from scipy import io
from sklearn.model_selection import train_test_split
from stein import SteinSampler
from stein.kernels import SquaredExponentialKernel
from stein.gradient_descent import Adagrad

# For reproducibility.
np.random.seed(0)

# Load data and partition into training and testing sets.
data = io.loadmat("./data/covertype.mat")["covtype"]
data_X, data_y = data[:, 1:], data[:, 0]
data_y[data_y == 2] = -1
X_train, X_test, y_train, y_test = train_test_split(
    data_X, data_y, test_size=0.2, random_state=0
)

# Size of minibatches during training.
n_batch = 50
# Number of training data points.
n_train = X_train.shape[0]

def grad_log_p(theta):
    """Gradient of the Bayesian logistic regression posterior distribution with
    respect to the linear coefficients as well as the precision over the
    coefficients.
    """
    # Select a minibatch on which to compute the gradient.
    batch_idx = np.random.choice(n_train, n_batch)
    X, y = X_train[batch_idx], y_train[batch_idx]
    # Extract logistic regression parameters.
    w = theta[:-1]
    alpha = np.exp(theta[-1])
    d = len(w)
    # Compute the linear component of logistic regression and compute the
    # probability of class one.
    linear = X.dot(w)
    p_hat = 1.0 / (1.0 + np.exp(-linear))
    # Gradient of the likelihood with respect to the coefficients.
    dw_data = ((y + 1) / 2.0 - p_hat).T.dot(X)
    # Gradient of the prior over coefficients.
    dw_prior = -alpha * w
    # Rescale gradient.
    dw = dw_data * X_train.shape[0] / X.shape[0] + dw_prior
    # Notice that we are performing gradient descent on the logarithm of the
    # prior precision. That's why this gradient is slightly different.
    dalpha = d / 2.0 - (alpha / 2.) * w.dot(w) - 0.01 * alpha + 1.

    return np.append(dw, dalpha)

def evaluation(theta, X_test, y_test):
    """Compute the test set accuracy of the Bayesian logistic regression
    algorithm using the posterior samples of the linear coefficients. Notice
    that we average class one probabilities over each particle before
    thresholding against one-half.
    """
    # Disregard the posterior precision over the linear coefficients.
    theta = theta[:, :-1]
    # Extract the number of particles and the number of testing data points.
    M = theta.shape[0]
    n_test = X_test.shape[0]
    # Compute the predictions from the Bayesian neural network.
    p_hat = 1. / (1. + np.exp(-X_test.dot(theta.T)))
    y_hat = (p_hat.mean(axis=1) > 0.5) * 2. - 1.

    return np.mean(y_hat == y_test)

# Create a function to serve as an evaluator in the Stein sampler class.
evaluator = partial(evaluation, X_test=X_test, y_test=y_test)

# Setup parameters of the Stein sampler.
n_particles = 100
n_params = X_train.shape[1] + 1
n_iters = 6000
# Specify that the Stein sampler should use a squared exponential kernel.
kernel = SquaredExponentialKernel(n_params)
# Create a gradient descent object for Stein variational gradient descent.
gd = Adagrad(learning_rate=1e-2)

# Create the Stein sampler.
stein = SteinSampler(grad_log_p, kernel, gd, evaluator=evaluator)
# Sample using Stein variational gradient descent with a squared exponential
# kernel on the posterior distribution over the parameters of a Bayesian
# logistic regression model.
theta = stein.sample(n_particles, n_iters)


# Compute the test set performance of the algorithm.
acc = evaluation(theta, X_test, y_test)
print("Accuracy:\t{}".format(acc))
