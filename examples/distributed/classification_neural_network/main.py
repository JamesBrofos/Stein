import numpy as np
import tensorflow as tf
from scipy import io
from sklearn.model_selection import train_test_split
from tensorflow.contrib.distributions import Normal, Gamma
from stein.samplers import DistributedSteinSampler
from stein.optimizers import AdamGradientDescent


# Load data and partition into training and testing sets.
data = io.loadmat("./data/applied/covertype.mat")["covtype"]
data_X, data_y = data[:, 1:], data[:, :1]
data_y[data_y == 2] = 0.
X_train, X_test, y_train, y_test = train_test_split(
    data_X, data_y, test_size=0.2
)
# Normalize the data. Notice that we do not normalize the test targets as this
# would result in non-comparable root mean squared errors.
X_train_mean, X_train_std = np.mean(X_train, 0), np.std(X_train, 0)
X_train = (X_train - X_train_mean) / X_train_std
X_test = (X_test - X_train_mean) / X_train_std

# Parameters for training such as the number of hidden neurons and the batch
# size to use during training, the total number of training iterations.
n_hidden = 50
n_batch = 100
# Number of training data points.
n_train, n_feats = X_train.shape
n_test = X_test.shape[0]
# Extract learning rate.
learning_rate = 1e-2
# Precision prior parameters.
alpha, beta = 1., 0.01

# Define a Bayesian neural network model in TensorFlow.
with tf.variable_scope("model"):
    # Placeholder variables for data.
    model_X = tf.placeholder(tf.float32, shape=[None, n_feats])
    model_y = tf.placeholder(tf.float32, shape=[None, 1])
    # TensorFlow variables for the networks weights and biases, the precision
    # over the weights and biases, and the noise precision of the output.
    model_log_lambda = tf.Variable(tf.zeros([]))
    model_lambda = tf.exp(model_log_lambda)
    model_w_1 = tf.Variable(tf.zeros([n_feats, n_hidden]))
    model_b_1 = tf.Variable(tf.zeros([n_hidden]))
    model_w_2 = tf.Variable(tf.zeros([n_hidden, 1]))
    model_b_2 = tf.Variable(tf.zeros([]))

    # Compute the prediction from the network.
    with tf.variable_scope("prediction"):
        logits = tf.matmul(
            tf.nn.relu(tf.matmul(model_X, model_w_1) + model_b_1), model_w_2
        ) + model_b_2
    # Likelihood function.
    with tf.variable_scope("likelihood"):
        log_l = -tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=model_y, logits=logits
        ))
    # Priors.
    with tf.variable_scope("priors"):
        prior_lambda = Gamma(alpha, beta)
        prior_w_1 = Normal(
            tf.zeros([n_feats, n_hidden]),
            tf.reciprocal(tf.sqrt(model_lambda))
        )
        prior_b_1 = Normal(
            tf.zeros([n_hidden]),
            tf.reciprocal(tf.sqrt(model_lambda))
        )
        prior_w_2 = Normal(
            tf.zeros([n_hidden, 1]),
            tf.reciprocal(tf.sqrt(model_lambda))
        )
        prior_b_2 = Normal(
            tf.zeros([]),
            tf.reciprocal(tf.sqrt(model_lambda))
        )
    # Compute the log-posterior distribution.
    log_p = (
        # Rescaled log-likelihood (to account for batch updates).
        log_l * n_train / n_batch +
        # Variance priors.
        prior_lambda.log_prob(model_lambda) +
        # Weight and bias priors.
        tf.reduce_sum(prior_w_1.log_prob(model_w_1)) +
        tf.reduce_sum(prior_w_2.log_prob(model_w_2)) +
        tf.reduce_sum(prior_b_1.log_prob(model_b_1)) +
        prior_b_2.log_prob(model_b_2)
    ) / n_train



# Gradient descent object.
gd = AdamGradientDescent(learning_rate=learning_rate, decay=0.999)
# Perform Stein variational gradient descent to sample from the posterior
# distribution of the Bayesian neural network.
n_particles = 20
n_threads = 4
sampler = DistributedSteinSampler(n_threads, n_particles, log_p, gd)


def evaluate(sampler, data_feed):
    """Evaluate the performance of the Bayesian neural network by computing its
    accuracy and log-likelihood on the test set.
    """
    # Average predictions across particles.
    logits_pred = sampler.function_posterior(logits, data_feed)
    avg_pred = logits_pred.mean(axis=0) > 0.
    acc = np.mean(avg_pred == y_test.ravel())
    ll = np.mean(sampler.function_posterior(log_l, data_feed)) / n_test
    # Evaluation.
    return acc, ll

# Current iteration of Stein variational gradient descent.
current_iter = 0
n_prog = 1
# Perform learning iterations.
while True:
    # Increment the global number of learning iterations.
    current_iter += 1
    # Train on batch.
    batch = np.random.choice(n_train, n_batch, replace=False)
    X, y = X_train[batch], y_train[batch]
    sampler.train_on_batch({model_X: X, model_y: y})

    # Output diagnostic variables.
    if current_iter % n_prog == 0:
        acc_test, ll_test = evaluate(
            sampler, {model_X: X_test, model_y: y_test}
        )
        print("Iteration {}:\t\t{:.4f}\t\t{:.4f}".format(
            current_iter, acc_test, ll_test
        ))
