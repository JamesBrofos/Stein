import numpy as np
import tensorflow as tf
from scipy import io
from sklearn.model_selection import train_test_split
from tensorflow.contrib.distributions import Normal, Gamma
from stein.samplers import SteinSampler
from stein.optimizers import AdamGradientDescent


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
        alpha_prior = Gamma(1., 0.01)
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
    accuracy on the test set.
    """
    # Average predictions across particles.
    logits_pred = sampler.function_posterior(logits, data_feed)
    # avg_pred = np.mean(1. / (1. + np.exp(-logits_pred)), axis=0) > 0.5
    avg_pred = logits_pred.mean(axis=0) > 0.
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

