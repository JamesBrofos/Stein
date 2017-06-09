import theano.tensor as T
import theano
import numpy as np
from scipy.spatial.distance import pdist, squareform
import random
import time
from stein.kernels import SquaredExponentialKernel
from stein import SteinSampler
from construct_neural_network import construct_neural_network

# seed = 5
# random.seed(seed)
# np.random.seed(seed)


''' load data file '''
data = np.loadtxt('./data/boston_housing')

# Please make sure that the last column is the label and the other columns are features
X_input = data[:, :-1]#range(data.shape[ 1 ] - 1) ]
y_input = data[:, -1]#data.shape[ 1 ] - 1 ]

# Prior variance variables.
alpha, beta = 1.0, 0.1


d = X_input.shape[1]   # number of data, dimension
batch_size, n_hidden, max_iter = 100, 50, 2000  # max_iter is a trade-off between running time and performance
num_vars = d * n_hidden + n_hidden * 2 + 3  # w1: d*n_hidden; b1: n_hidden; w2 = n_hidden; b2 = 1; 2 variances

nn_predict, logp_gradient = construct_neural_network(num_vars, alpha, beta)


'''
    Sample code to reproduce our results for the Bayesian neural network example.
    Our settings are almost the same as Hernandez-Lobato and Adams (ICML15) https://jmhldotorg.files.wordpress.com/2015/05/pbp-icml2015.pdf
    Our implementation is also based on their Python code.

    p(y | W, X, \gamma) = \prod_i^N  N(y_i | f(x_i; W), \gamma^{-1})
    p(W | \lambda) = \prod_i N(w_i | 0, \lambda^{-1})
    p(\gamma) = Gamma(\gamma | a0, b0)
    p(\lambda) = Gamma(\lambda | a0, b0)

    The posterior distribution is as follows:
    p(W, \gamma, \lambda) = p(y | W, X, \gamma) p(W | \lambda) p(\gamma) p(\lambda)
    To avoid negative values of \gamma and \lambda, we update loggamma and loglambda instead.

    Copyright (c) 2016,  Qiang Liu & Dilin Wang
    All rights reserved.
'''

'''
We define a one-hidden-layer-neural-network specifically. We leave extension of deep neural network as our future work.

Input
-- X_train: training dataset, features
-- y_train: training labels
-- batch_size: sub-sampling batch size
-- max_iter: maximum iterations for the training procedure
-- M: number of particles are used to fit the posterior distribution
-- n_hidden: number of hidden units
-- a0, b0: hyper-parameters of Gamma distribution
-- master_stepsize, auto_corr: parameters of adgrad
'''
def train(
        X_train,
        y_train,
        batch_size=100,
        max_iter=1000,
        M=20,
        n_hidden=50,
):
    theta = np.zeros([M, num_vars])  # particles, will be initialized later

    '''
    We keep the last 10% (maximum 500) of training data points for model developing
    '''
    size_dev = min(int(np.round(0.1 * X_train.shape[0])), 500)
    X_dev, y_dev = X_train[-size_dev:], y_train[-size_dev:]
    X_train, y_train = X_train[:-size_dev], y_train[:-size_dev]


    '''
    Training with SVGD
    '''
    # normalization
    X_train, y_train = normalization(X_train, y_train)
    N0 = X_train.shape[0]  # number of observations

    ''' initializing all particles '''
    for i in range(M):
        w1, b1, w2, b2, loggamma, loglambda = init_weights(alpha, beta)
        # use better initialization for gamma
        ridx = np.random.choice(range(X_train.shape[0]), \
                                np.min([X_train.shape[0], 1000]), replace = False)
        y_hat = nn_predict(X_train[ridx,:], w1, b1, w2, b2)
        loggamma = -np.log(np.mean(np.power(y_hat - y_train[ridx], 2)))
        theta[i] = pack_weights(w1, b1, w2, b2, loggamma, loglambda)

    def grad_log_p(theta):
        batch = np.random.choice(N0, batch_size)
        w1, b1, w2, b2, loggamma, loglambda = unpack_weights(theta)
        dw1, db1, dw2, db2, dloggamma, dloglambda = logp_gradient(X_train[batch], y_train[batch], w1, b1, w2, b2, loggamma, loglambda, N0)
        return pack_weights(dw1, db1, dw2, db2, dloggamma, dloglambda)

    kernel = SquaredExponentialKernel(num_vars)
    stein = SteinSampler(grad_log_p, kernel)
    theta = stein.sample(M, max_iter, learning_rate=1e-3, theta_init=theta)

    '''
    Model selection by using a development set
    '''
    X_dev = normalization(X_dev)
    for i in range(M):
        w1, b1, w2, b2, loggamma, loglambda = unpack_weights(theta[i, :])
        pred_y_dev = nn_predict(X_dev, w1, b1, w2, b2) * std_y_train + mean_y_train
        # likelihood
        def f_log_lik(loggamma):
            return np.sum(  np.log(np.sqrt(np.exp(loggamma)) /np.sqrt(2*np.pi) * np.exp( -1 * (np.power(pred_y_dev - y_dev, 2) / 2) * np.exp(loggamma) )) )
        # The higher probability is better
        lik1 = f_log_lik(loggamma)
        # one heuristic setting
        loggamma = -np.log(np.mean(np.power(pred_y_dev - y_dev, 2)))
        lik2 = f_log_lik(loggamma)
        if lik2 > lik1:
            theta[i,-2] = loggamma  # update loggamma

    return theta, nn_predict

def normalization(X, y = None):
    X = (X - np.full(X.shape, mean_X_train)) / \
        np.full(X.shape, std_X_train)

    if y is not None:
        y = (y - mean_y_train) / std_y_train
        return (X, y)
    else:
        return X

'''
Initialize all particles
'''
def init_weights(a0, b0):
    w1 = 1.0 / np.sqrt(d + 1) * np.random.randn(d, n_hidden)
    b1 = np.zeros((n_hidden,))
    w2 = 1.0 / np.sqrt(n_hidden + 1) * np.random.randn(n_hidden)
    b2 = 0.
    loggamma = np.log(np.random.gamma(a0, b0))
    loglambda = np.log(np.random.gamma(a0, b0))
    return (w1, b1, w2, b2, loggamma, loglambda)


'''
Pack all parameters in our model
'''
def pack_weights(w1, b1, w2, b2, loggamma, loglambda):
    params = np.concatenate([w1.flatten(), b1, w2, [b2], [loggamma],[loglambda]])
    return params

'''
Unpack all parameters in our model
'''
def unpack_weights(z):
    w = z
    w1 = np.reshape(w[:d*n_hidden], [d, n_hidden])
    b1 = w[d*n_hidden:(d+1)*n_hidden]
    
    w = w[(d+1)*n_hidden:]
    w2, b2 = w[:n_hidden], w[-3]
    
    # the last two parameters are log variance
    loggamma, loglambda= w[-2], w[-1]

    return (w1, b1, w2, b2, loggamma, loglambda)


'''
Evaluating testing rmse and log-likelihood, which is the same as in PBP
Input:
-- X_test: unnormalized testing feature set
-- y_test: unnormalized testing labels
'''
def evaluation(theta, X_test, y_test, nn_predict):
    M = theta.shape[0]
    # normalization
    X_test = normalization(X_test)
    
    # average over the output
    pred_y_test = np.zeros([M, len(y_test)])
    prob = np.zeros([M, len(y_test)])
        
    '''
    Since we have M particles, we use a Bayesian view to calculate rmse and log-likelihood
    '''
    for i in range(M):
        w1, b1, w2, b2, loggamma, loglambda = unpack_weights(theta[i, :])
        pred_y_test[i, :] = nn_predict(X_test, w1, b1, w2, b2) * std_y_train + mean_y_train
        prob[i, :] = np.sqrt(np.exp(loggamma)) /np.sqrt(2*np.pi) * np.exp( -1 * (np.power(pred_y_test[i, :] - y_test, 2) / 2) * np.exp(loggamma) )
    pred = np.mean(pred_y_test, axis=0)

    # evaluation
    svgd_rmse = np.sqrt(np.mean((pred - y_test)**2))
    svgd_ll = np.mean(np.log(np.mean(prob, axis = 0)))

    return (svgd_rmse, svgd_ll)


''' build the training and testing data set'''
train_ratio = 0.9 # We create the train and test sets with 90% and 10% of the data
permutation = np.arange(X_input.shape[0])
random.shuffle(permutation)

size_train = int(np.round(X_input.shape[ 0 ] * train_ratio))
index_train = permutation[ 0 : size_train]
index_test = permutation[ size_train : ]

X_train, y_train = X_input[ index_train, : ], y_input[ index_train ]
X_test, y_test = X_input[ index_test, : ], y_input[ index_test ]

std_X_train = np.std(X_train, 0)
std_X_train[ std_X_train == 0 ] = 1
mean_X_train = np.mean(X_train, 0)

mean_y_train = np.mean(y_train)
std_y_train = np.std(y_train)


start = time.time()
''' Training Bayesian neural network with SVGD '''
theta, nn_predict = train(X_train, y_train, batch_size = batch_size, n_hidden = n_hidden, max_iter = max_iter)
svgd_time = time.time() - start
svgd_rmse, svgd_ll = evaluation(theta, X_test, y_test, nn_predict)
print('SVGD', svgd_rmse, svgd_ll, svgd_time)
