import numpy as np
from stein.samplers import SteinSampler
from stein.gradient_descent import AdamGradientDescent
from model_and_data import (
    log_p,
    n_train,
    n_batch,
    logits,
    model_X,
    model_y,
    X_train,
    y_train,
    X_test,
    y_test
)


def evaluate(sampler, data_feed):
    """Evaluate the performance of the Bayesian neural network by computing its
    root mean squared error and log-likelihood on a test dataset. Notice that
    this function performs normalization based on the mean and variance
    parameters of the training data.
    """
    # Construct vectors to store the prediction for each of the particles and
    # each of the test data points under the posterior.
    logits_pred = np.zeros([n_particles, data_feed[model_y].shape[0]])
    # We adopt a Bayesian perspective on computing the metrics by averaging over
    # the predictions of each constituent particle.
    for i in range(n_particles):
        feed = {v: x[i] for v, x in sampler.theta.items()}
        feed.update(data_feed)
        logits_pred[i] = sampler.sess.run(logits, feed).ravel()
    # Average predictions across particles.
    avg_pred = np.mean(1. / (1. + np.exp(-logits_pred)), axis=0) > 0.5
    # Evaluation.
    return np.mean(avg_pred == y_test.ravel())


# Number of learning iterations.
n_iters = 6000
n_prog = 100
# Sample from the posterior using Stein variational gradient descent.
n_particles = 100
gd = AdamGradientDescent(learning_rate=1e-1)
sampler = SteinSampler(n_particles, log_p, gd)
# Perform learning iterations.
for i in range(n_iters):
    if i % n_prog == 0:
        acc = evaluate(sampler, {model_X: X_test, model_y: y_test})
        print("Iteration {} / {}: {:4f}".format(i, n_iters, acc))
    # Train on batch.
    batch = np.random.choice(n_train, n_batch, replace=False)
    X, y = X_train[batch], y_train[batch]
    sampler.train_on_batch({model_X: X, model_y: y})

