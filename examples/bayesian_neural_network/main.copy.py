import numpy as np
from functools import partial
from sklearn.model_selection import train_test_split
from stein.kernels import SquaredExponentialKernel
from stein import SteinSampler
from construct_neural_network import construct_neural_network


# For reproducibility.
np.random.seed(0)

# Load data and partition into training and testing sets.
data = np.loadtxt("./data/boston_housing")
data_X, data_y = data[:, :-1], data[:, -1]
X_train, X_test, y_train, y_test = train_test_split(
    data_X, data_y, test_size=0.1, random_state=0
)
# Normalize the data.
mean_X_train, std_X_train = np.mean(X_train, axis=0), np.std(X_train, axis=0)
mean_y_train, std_y_train = np.mean(y_train), np.std(y_train)

def normalize(X, mean, std):
    return (X - mean) / std

X_train = normalize(X_train, mean_X_train, std_X_train)
y_train = normalize(y_train, mean_y_train, std_y_train)

# Size of minibatches during training.
n_batch = 100
# Number of training data points.
n_train, n_vars = X_train.shape

# Number of hidden units per layer.
n_hidden = 50
n_params = n_vars * n_hidden + n_hidden * 2 + 3

# Methods to compute the prediction and the gradient of the log-posterior.
prediction_func, grad_log_p_func = construct_neural_network(n_params)

def grad_log_p(theta):
    """Computes the gradient of a Bayesian neural network with respect to the
    weights and biases in addition to two variance parameters. The Bayesian
    model can be expressed as follows:

        p(y | W, X, \gamma) = \prod_i^N  N(y_i | f(x_i; W), \gamma^{-1})
        p(W | \lambda) = \prod_i N(w_i | 0, \lambda^{-1})
        p(\gamma) = Gamma(\gamma | alpha, beta)
        p(\lambda) = Gamma(\lambda | alpha, beta)

    In this code, we'll implement a single layer neural network.
    """
    # Select a minibatch on which to compute the gradient.
    batch_idx = np.random.choice(n_train, n_batch)
    X, y = X_train[batch_idx], y_train[batch_idx]
    w1, b1, w2, b2, loggamma, loglambda = unpack_weights(theta)
    return pack_weights(
        *grad_log_p_func(X, y, w1, b1, w2, b2, loggamma, loglambda, n_train)
    )

def pack_weights(w1, b1, w2, b2, loggamma, loglambda):
    return np.concatenate(
        [w1.flatten(), b1, w2, [b2], [loggamma], [loglambda]]
    )

def unpack_weights(theta):
    w1 = np.reshape(theta[:n_vars*n_hidden], [n_vars, n_hidden])
    b1 = theta[n_vars*n_hidden:(n_vars+1)*n_hidden]
    theta = theta[(n_vars+1)*n_hidden:]
    w2, b2 = theta[:n_hidden], theta[-3]
    # The last two parameters are log variance.
    loggamma, loglambda= theta[-2], theta[-1]

    return w1, b1, w2, b2, loggamma, loglambda

def initialize_particles():
    w1 = 1.0 / np.sqrt(n_vars + 1) * np.random.randn(n_vars, n_hidden)
    b1 = np.zeros((n_hidden,))
    w2 = 1.0 / np.sqrt(n_hidden + 1) * np.random.randn(n_hidden)
    b2 = 0.
    loggamma = np.log(np.random.gamma(1., 0.1))
    loglambda = np.log(np.random.gamma(1., 0.1))
    return w1, b1, w2, b2, loggamma, loglambda

# Setup parameters of the Stein sampler.
n_particles = 20
n_iters = 2000
theta = np.zeros((n_particles, n_params))
for i in range(n_particles):
    w1, b1, w2, b2, loggamma, loglambda = initialize_particles()
    # Use better initialization for gamma.
    ridx = np.random.choice(range(X_train.shape[0]), np.min([X_train.shape[0], 1000]), replace=False)
    y_hat = prediction_func(X_train[ridx,:], w1, b1, w2, b2)
    loggamma = -np.log(np.mean(np.power(y_hat - y_train[ridx], 2)))
    theta[i,:] = pack_weights(w1, b1, w2, b2, loggamma, loglambda)

# Specify that the Stein sampler should use a squared exponential kernel.
kernel = SquaredExponentialKernel(n_params)

def evaluation(theta, X_test, y_test):
    # Normalization
    X_test = normalize(X_test, mean_X_train, std_X_train)
    # Average over the output.
    pred_y_test = np.zeros([n_particles, len(y_test)])
    # Average of all of the particles.
    for i in range(n_particles):
        w1, b1, w2, b2, loggamma, loglambda = unpack_weights(theta[i])
        pred_y_test[i] = prediction_func(X_test, w1, b1, w2, b2) * std_y_train + mean_y_train

    pred = np.mean(pred_y_test, axis=0)
    return np.sqrt(np.mean((pred - y_test)**2))

eval_ = partial(evaluation, X_test=X_test, y_test=y_test)

# Create the Stein sampler.
stein = SteinSampler(grad_log_p, kernel)
# Sample using Stein variational gradient descent with a squared exponential
# kernel on the posterior distribution over the parameters of a Bayesian
# logistic regression model.
theta = stein.sample(
    n_particles, n_iters, learning_rate=1e-3, theta=theta, evaluation=eval_
)


mse = evaluation(theta, X_test, y_test)
print("Mean squared error:\t{}".format(mse))


X_test = np.loadtxt("X_test.original.txt")
y_test = np.loadtxt("y_test.original.txt")
theta = np.loadtxt("theta.original.txt")
mse = evaluation(theta, X_test, y_test)
print("Mean squared error:\t{}".format(mse))
