import numpy as np
from stein.samplers import SteinSampler
from stein.gradient_descent import AdamGradientDescent, AdagradGradientDescent
from model_and_data import (
    n_particles,
    n_train,
    n_batch,
    learning_rate,
    log_p,
    log_l,
    pred,
    theta,
    model_X,
    model_y,
    model_log_gamma,
    y_train_mean,
    y_train_std,
    X_train,
    X_dev,
    X_test,
    y_train,
    y_dev,
    y_test
)


# Set interval at which to record output.
n_prog = 100

def evaluate(sampler, data_feed):
    """Evaluate the performance of the Bayesian neural network by computing its
    root mean squared error and log-likelihood on a test dataset. Notice that
    this function performs normalization based on the mean and variance
    parameters of the training data.
    """
    # We adopt a Bayesian perspective on computing the metrics by averaging over
    # the predictions of each constituent particle.
    y_test_pred = (
        sampler.function_posterior(pred, data_feed) * y_train_std + y_train_mean
    )
    # Average predictions across particles.
    avg_pred = np.mean(y_test_pred, axis=0)
    # Evaluation.
    rmse = np.sqrt(np.mean((avg_pred - data_feed[model_y].ravel())**2))

    # We also want to compute the test log-likelihood to understand the
    # probability of the data under the posterior model. However, after applying
    # heuristics to the posterior prediction precision, we'll reset the value to
    # its original state.
    log_gamma_copy = sampler.theta[model_log_gamma].copy()
    # Prediction on the development set, making sure to project the prediction
    # into the real target space by unnormalizing.
    y_dev_pred = (
        sampler.function_posterior(
            pred,
            {model_X: X_dev, model_y: y_dev}
        )
    ) * y_train_std + y_train_mean
    # Apply a heuristic for the prediction precision that tends to increase test
    # set likelihood.
    for i in range(sampler.n_particles):
        y_pred = y_dev_pred[i].ravel()
        sampler.theta[model_log_gamma][i] = -np.log(np.mean(
            (y_pred - (y_dev.ravel() * y_train_std + y_train_mean)) ** 2
        ))

    # Create a matrix to store the probability of each target variable in the
    # development set under the posterior model.
    n_test = data_feed[model_X].shape[0]
    prob = np.zeros([sampler.n_particles, n_test])
    # Iterate over each particle and compute the log-likelihood of the test
    # data.
    for i in range(sampler.n_particles):
        prob[i] = np.sqrt(
            np.exp(sampler.theta[model_log_gamma][i])
        ) / np.sqrt(2*np.pi) * np.exp(
            -0.5 * (y_test_pred[i] - data_feed[model_y].ravel()) ** 2 *
            np.exp(sampler.theta[model_log_gamma][i])
        )
    # Compute average log-likelihood.
    ll = np.mean(np.log(np.mean(prob, axis=0)))
    # Reset.
    sampler.theta[model_log_gamma] = log_gamma_copy

    return rmse, ll

# Current iteration of Stein variational gradient descent.
current_iter = 0
# Gradient descent object.
gd = AdamGradientDescent(learning_rate=learning_rate, decay=0.999)
# Perform Stein variational gradient descent to sample from the posterior
# distribution of the Bayesian neural network.
sampler = SteinSampler(n_particles, log_p, gd, theta)

while True:
    # Increment the global number of learning iterations.
    current_iter += 1
    # Train on batch.
    batch = np.random.choice(n_train, n_batch, replace=False)
    X, y = X_train[batch], y_train[batch]
    sampler.train_on_batch({model_X: X, model_y: y})

    # Output diagnostic variables.
    if current_iter % n_prog == 0:
        rmse_train, ll_train = evaluate(sampler, {
            model_X: X_train,
            model_y: y_train * y_train_std + y_train_mean
        })
        rmse_test, ll_test = evaluate(
            sampler, {model_X: X_test, model_y: y_test}
        )
        print("Iteration {}:\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}".format(
            current_iter, rmse_train, rmse_test, ll_test
        ))

