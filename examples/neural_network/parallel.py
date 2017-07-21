import numpy as np
from stein.samplers import ParallelSteinSampler
from stein.gradient_descent import AdamGradientDescent, AdagradGradientDescent
from model_and_data import (
    n_particles,
    n_train,
    n_batch,
    log_p,
    pred,
    theta,
    model_X,
    model_y,
    y_train_mean,
    y_train_std,
    X_train,
    X_test,
    y_train,
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
    # Merge the particles.
    theta = sampler.merge()

    if sampler.comm.rank == 0:
        # Construct vectors to store the prediction for each of the particles
        # and each of the test data points under the posterior.
        y_test_pred = np.zeros([n_particles, data_feed[model_y].shape[0]])
        # We adopt a Bayesian perspective on computing the metrics by averaging
        # over the predictions of each constituent particle.
        for i in range(n_particles):
            feed = {v: x[i] for v, x in theta.items()}
            feed.update(data_feed)
            y_test_pred[i] = (sampler.sampler.sess.run(
                pred, feed
            ) * y_train_std + y_train_mean).ravel()

        # Average predictions across particles.
        avg_pred = np.mean(y_test_pred, axis=0)
        # Evaluation.
        rmse = np.sqrt(np.mean((avg_pred - data_feed[model_y].ravel())**2))

        return rmse

# Current iteration of Stein variational gradient descent.
current_iter = 0
# Gradient descent object.
gd = AdamGradientDescent(learning_rate=1e-1)
# Perform Stein variational gradient descent to sample from the posterior
# distribution of the Bayesian neural network.
sampler = ParallelSteinSampler(n_particles, log_p, gd, theta)

while True:
    # Increment the global number of learning iterations.
    current_iter += 1
    # Train on batch.
    batch = np.random.choice(n_train, n_batch, replace=False)
    X, y = X_train[batch], y_train[batch]
    sampler.train_on_batch({model_X: X, model_y: y})
    if current_iter % 10 == 0:
        sampler.shuffle()

    # Output diagnostic variables.
    if current_iter % n_prog == 0:
        rmse_train = evaluate(sampler, {
            model_X: X_train,
            model_y: y_train * y_train_std + y_train_mean
        })
        rmse_test = evaluate(sampler, {model_X: X_test, model_y: y_test})
        if sampler.comm.rank == 0:
            print("Iteration {}:\t\t{:.4f}\t\t{:.4f}".format(
                current_iter, rmse_train, rmse_test
            ))
