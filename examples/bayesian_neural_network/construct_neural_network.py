import numpy as np
import theano.tensor as T
import theano


def construct_neural_network(n_params, alpha, beta, nonlin=T.nnet.relu):
    # Data variables.
    X = T.matrix("X") # Feature matrix.
    y = T.vector("y") # Labels.
    # Network parameters.
    w_1 = T.matrix("w_1") # Weights between input layer and hidden layer.
    b_1 = T.vector("b_1") # Bias vector of hidden layer.
    w_2 = T.vector("w_2") # Weights between hidden layer and output layer.
    b_2 = T.scalar("b_2") # Bias of output.
    # Number of observations.
    N = T.scalar("N")
    # Variances parameters.
    log_gamma = T.scalar("log_gamma")
    log_lambda = T.scalar("log_lambda")

    # Prediction produced by the neural network.
    prediction = T.dot(nonlin(T.dot(X, w_1) + b_1), w_2) + b_2
    # Define the components of the log-posterior distribution.
    log_likelihood = (
        -0.5 * X.shape[0] * (T.log(2*np.pi) - log_gamma)
        - (T.exp(log_gamma)/2) * T.sum(T.power(prediction - y, 2))
    )
    log_prior_data = (
        (alpha - 1.) * log_gamma - beta * T.exp(log_gamma) + log_gamma
    )
    log_prior_w = (
        -0.5 * (n_params-2) * (T.log(2*np.pi)-log_lambda)
        - (T.exp(log_lambda)/2)*(
            (w_1**2).sum() + (w_2**2).sum() + (b_1**2).sum() + b_2**2
        ) + (alpha - 1.) * log_lambda - beta * T.exp(log_lambda) + log_lambda
    )

    # Rescaled log-posterior because we're using minibatches.
    log_posterior = (
        log_likelihood * N / X.shape[0] + log_prior_data + log_prior_w
    )
    dw_1, db_1, dw_2, db_2, d_log_gamma, d_log_lambda = T.grad(
        log_posterior, [w_1, b_1, w_2, b_2, log_gamma, log_lambda]
    )

    # Compute the prediction of the neural network.
    prediction_func = theano.function(
        inputs=[X, w_1, b_1, w_2, b_2], outputs=prediction
    )
    # Compute the gradient.
    grad_log_p_func = theano.function(
        inputs=[X, y, w_1, b_1, w_2, b_2, log_gamma, log_lambda, N],
        outputs=[dw_1, db_1, dw_2, db_2, d_log_gamma, d_log_lambda]
    )

    return prediction_func, grad_log_p_func
