import numpy as np
import tensorflow as tf
from tensorflow.contrib.distributions import Normal


# Import data.
data_X = np.loadtxt("./data/data_X.csv", delimiter=",")
if len(data_X.shape) == 1:
    data_X = np.atleast_2d(data_X).T
data_w = np.atleast_2d(np.loadtxt("./data/data_w.csv", delimiter=",")).T
data_y = np.atleast_2d(np.loadtxt("./data/data_y.csv", delimiter=",")).T
n_samples, n_feats = data_X.shape


with tf.variable_scope("model"):
    # Placeholders for features and targets.
    model_X = tf.placeholder(tf.float32, shape=[None, n_feats])
    model_y = tf.placeholder(tf.float32, shape=[None, 1])
    model_w = tf.Variable(tf.zeros([n_feats, 1]))
    # Compute prior.
    with tf.variable_scope("priors"):
        w_prior = Normal(tf.zeros([n_feats, 1]), 1.)
    # Compute likelihood function.
    with tf.variable_scope("likelihood"):
        y_hat = tf.matmul(model_X, model_w)
        log_l = -0.5 * tf.reduce_sum(tf.square(y_hat - model_y))
    # Compute the log-posterior of the model.
    log_p = log_l + tf.reduce_sum(w_prior.log_prob(model_w))
