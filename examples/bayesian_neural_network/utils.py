import numpy as np


def pack_weights(w1, b1, w2, b2, loggamma, loglambda):
    """Convert the weights, biases, and variance parameters of a Bayesian neural
    network into a vector representation.
    """
    params = np.concatenate([w1.flatten(), b1, w2, [b2], [loggamma],[loglambda]])
    return params

def unpack_weights(theta, n_vars, n_hidden):
    """Convert a vector representation of the weights, biases, and variance
    parameters of a neural network into an intelligible representation.
    """
    w = theta
    w1 = np.reshape(w[:n_vars*n_hidden], [n_vars, n_hidden])
    b1 = w[n_vars*n_hidden:(n_vars+1)*n_hidden]
    w = w[(n_vars+1)*n_hidden:]
    w2, b2 = w[:n_hidden], w[-3]
    loggamma, loglambda= w[-2], w[-1]

    return w1, b1, w2, b2, loggamma, loglambda

def normalization(X_and_mean_and_variance, y_and_mean_and_variance=None):
    """Normalize the input by subtracting the mean and dividing by the standard
    deviation. This ensures that each column of the input has zero mean and unit
    variance.
    """
    X = (
        (X_and_mean_and_variance[0] - X_and_mean_and_variance[1]) /
        X_and_mean_and_variance[2]
    )

    if y_and_mean_and_variance is not None:
        y = (
            (y_and_mean_and_variance[0] - y_and_mean_and_variance[1]) /
            y_and_mean_and_variance[2]
        )
        return X, y
    else:
        return X

def init_weights(alpha, beta, n_vars, n_hidden):
    """Sample from the prior distribution over weights and biases of the
    network. The variance parameters are also sampled from a prior distribution.
    """
    w1 = 1.0 / np.sqrt(n_vars + 1) * np.random.randn(n_vars, n_hidden)
    b1 = np.zeros((n_hidden,))
    w2 = 1.0 / np.sqrt(n_hidden + 1) * np.random.randn(n_hidden)
    b2 = 0.
    loglambda = np.log(np.random.gamma(alpha, beta))
    return w1, b1, w2, b2, loglambda
