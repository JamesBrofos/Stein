import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.contrib.distributions import Normal, Gamma
from stein import SteinSampler
from stein.gradient_descent import AdamGradientDescent


# For reproducibility.
np.random.seed(0)
tf.set_random_seed(0)

# Generate random data from a logistic regression model.
n_samples, n_feats = 1000, 1
data_X = np.random.normal(size=(n_samples, n_feats))
data_w = np.random.normal(scale=3., size=(n_feats, 1))
data_p = 1. / (1. + np.exp(-data_X.dot(data_w)))
data_y = np.random.binomial(1, data_p)

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
    # Compute the gradient of the log-posterior with respect to the model linear
    # coefficients.
    log_p = (
        log_l +
        tf.reduce_sum(w_prior.log_prob(model_w)) +
        alpha_prior.log_prob(model_alpha)
    )


# Number of learning iterations.
n_iters = 1000
n_prog = n_iters // 10
# Sample from the posterior using Stein variational gradient descent.
n_particles = 50
gd = AdamGradientDescent(learning_rate=1e-1)
sampler = SteinSampler(n_particles, log_p, gd)
# Perform learning iterations.
for i in range(n_iters):
    if i % n_prog == 0:
        print("Iteration: {} / {}".format(i, n_iters))
    sampler.train_on_batch({model_X: data_X, model_y: data_y})


# Visualize if there is only a single dimension.
if n_feats == 1:
    X_plot = np.atleast_2d(np.linspace(-3., 3., num=100)).T
    plt.plot(data_X.ravel(), data_y.ravel(), "r.")
    for i in range(n_particles):
        p_plot = 1. / (1. + np.exp(-X_plot.dot(sampler.theta[model_w][i])))
        plt.plot(X_plot.ravel(), p_plot.ravel(), "g-", alpha=0.1)
    p = 1. / (1. + np.exp(-X_plot.dot(data_w)))
    plt.plot(X_plot.ravel(), p.ravel(), "b-", linewidth=2.)
    plt.grid()
    plt.show()
