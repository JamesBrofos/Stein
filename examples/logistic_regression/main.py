import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import io
from sklearn.model_selection import train_test_split
from tensorflow.contrib.distributions import Normal, Gamma
from stein.samplers import SteinSampler
from stein.gradient_descent import AdamGradientDescent


# For reproducibility.
# np.random.seed(0)

# Load data and partition into training and testing sets.
data = io.loadmat("./data/covertype.mat")["covtype"]
data_X, data_y = data[:, 1:], data[:, :1]
data_y[data_y == 2] = 0.
X_train, X_test, y_train, y_test = train_test_split(
    data_X, data_y, test_size=0.2
)
# Size of minibatches during training.
n_batch = 50
# Number of training data points.
n_train, n_feats = X_train.shape

# Define a logistic regression model in TensorFlow.
with tf.variable_scope("model"):
    # Placeholders include features, labels, linear coefficients, and prior
    # covariance.
    model_X = tf.placeholder(tf.float32, shape=[None, n_feats])
    model_y = tf.placeholder(tf.float32, shape=[None, 1])
    model_w = tf.Variable(tf.zeros([n_feats, 1]))
    model_log_alpha = tf.Variable(tf.zeros([]))
    model_alpha = tf.exp(model_log_alpha)
    # Compute prior.
    with tf.variable_scope("priors"):
        w_prior = Normal(
            tf.zeros([n_feats, 1]),
            tf.reciprocal(tf.sqrt(model_alpha))
        )
        alpha_prior = Gamma(concentration=1., rate=0.01)
    # Compute the likelihood function.
    with tf.variable_scope("likelihood"):
        logits = tf.matmul(model_X, model_w)
        log_l = -tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=model_y, logits=logits
        ))
    # Compute the log-posterior of the model.
    log_p = (
        log_l * (n_train / n_batch) +
        tf.reduce_sum(w_prior.log_prob(model_w)) +
        alpha_prior.log_prob(model_alpha)
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
    batch = np.random.choice(n_train, n_batch, replace=False)
    X, y = X_train[batch], y_train[batch]
    sampler.train_on_batch({model_X: X, model_y: y})

