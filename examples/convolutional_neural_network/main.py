import numpy as np
import tensorflow as tf
from tensorflow.contrib.distributions import Normal, Gamma
from tensorflow.examples.tutorials.mnist import input_data
from stein import SteinSampler
from stein.gradient_descent import AdamGradientDescent


# Load MNIST data.
mnist = input_data.read_data_sets("data", one_hot=True)
# Batch size, the number of learning iterations before progress is logged.
n_batch = 50
n_prog = 100
n_particles = 10
n_train = mnist.train.images.shape[0]
# Precision prior parameters.
alpha, beta = 1., 0.01

# Define the neural network model.
with tf.variable_scope("model"):
    # Placeholder variables for images and labels.
    model_X = tf.placeholder(tf.float32, shape=[None, 784])
    model_y = tf.placeholder(tf.float32, shape=[None, 10])
    model_X_image = tf.reshape(model_X, [-1, 28, 28, 1])
    # TensorFlow variables for convolutional weights and biases, fully connected
    # weights and biases.
    model_log_lambda = tf.Variable(tf.zeros([]))
    model_log_gamma = tf.Variable(tf.zeros([]))
    model_lambda = tf.exp(model_log_lambda)
    model_gamma = tf.exp(model_log_gamma)
    model_W_conv_1 = tf.Variable(tf.zeros([8, 8, 1, 16]))
    model_W_conv_2 = tf.Variable(tf.zeros([4, 4, 16, 32]))
    model_b_conv_1 = tf.Variable(tf.zeros([16]))
    model_b_conv_2 = tf.Variable(tf.zeros([32]))
    model_W_1 = tf.Variable(tf.zeros([4*4*32, 1024]))
    model_W_2 = tf.Variable(tf.zeros([1024, 10]))
    model_b_1 = tf.Variable(tf.zeros([1024]))
    model_b_2 = tf.Variable(tf.zeros([10]))

    # Model predictions in logit form.
    with tf.variable_scope("prediction"):
        conv_1 = tf.nn.relu(tf.nn.conv2d(
            model_X_image, model_W_conv_1, strides=[1, 4, 4, 1], padding="SAME"
        ) + model_b_conv_1)
        conv_2 = tf.nn.relu(tf.nn.conv2d(
            conv_1, model_W_conv_2, strides=[1, 2, 2, 1], padding="SAME"
        ) + model_b_conv_2)
        logits = tf.matmul(tf.nn.relu(tf.matmul(
            tf.reshape(conv_2, [-1, 4*4*32]), model_W_1
        ) + model_b_1), model_W_2) + model_b_2
    # Define priors.
    with tf.variable_scope("priors"):
        prior_gamma = Gamma(alpha, beta)
        prior_lambda = Gamma(alpha, beta)
        prior_W_conv_1 = Normal(
            tf.zeros([8, 8, 1, 16]), tf.reciprocal(tf.sqrt(model_lambda))
        )
        prior_b_conv_1 = Normal(
            tf.zeros([16]), tf.reciprocal(tf.sqrt(model_lambda))
        )
        prior_W_conv_2 = Normal(
            tf.zeros([4, 4, 16, 32]), tf.reciprocal(tf.sqrt(model_lambda))
        )
        prior_b_conv_2 = Normal(
            tf.zeros([32]), tf.reciprocal(tf.sqrt(model_lambda))
        )
        prior_W_1 = Normal(
            tf.zeros([4*4*32, 1024]), tf.reciprocal(tf.sqrt(model_gamma))
        )
        prior_b_1 = Normal(
            tf.zeros([1024]), tf.reciprocal(tf.sqrt(model_gamma))
        )
        prior_W_2 = Normal(
            tf.zeros([1024, 10]), tf.reciprocal(tf.sqrt(model_gamma))
        )
        prior_b_2 = Normal(
            tf.zeros([10]), tf.reciprocal(tf.sqrt(model_gamma))
        )

    # Model likelihood function.
    with tf.variable_scope("likelihood"):
        log_l = -tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=model_y, logits=logits
        ))
    # Compute the log-posterior distribution.
    log_p = (
        # Rescaled log-likelihood (to account for batch updates).
        log_l * n_train / n_batch +
        # Variance priors.
        prior_gamma.log_prob(model_gamma) +
        prior_lambda.log_prob(model_lambda) +
        # Weight and biases priors.
        tf.reduce_sum(prior_W_conv_1.log_prob(model_W_conv_1)) +
        tf.reduce_sum(prior_W_conv_2.log_prob(model_W_conv_2)) +
        tf.reduce_sum(prior_b_conv_1.log_prob(model_b_conv_1)) +
        tf.reduce_sum(prior_b_conv_2.log_prob(model_b_conv_2)) +
        tf.reduce_sum(prior_W_1.log_prob(model_W_1)) +
        tf.reduce_sum(prior_W_2.log_prob(model_W_2)) +
        tf.reduce_sum(prior_b_1.log_prob(model_b_1)) +
        tf.reduce_sum(prior_b_2.log_prob(model_b_2))
    )

# Create a function to measure accuracy.
def evaluation(sampler, data_feed):
    """Compute the test set accuracy of the Bayesian convolutional neural
    network algorithm using the posterior samples of the weights and biases.
    Notice that we average logits over each particle and then compute the index
    of the maximum.
    """
    # Construct a vector to hold the logits for each sampled particle.
    P = np.zeros([n_particles, data_feed[model_y].shape[0], 10])
    for i in range(n_particles):
        feed = {v: x[i] for v, x in sampler.theta.items()}
        P[i] = sampler.sess.run(logits, {**feed, **data_feed})

    a = np.mean(np.argmax(P.mean(axis=0), 1) == np.argmax(data_feed[model_y], 1))
    return a

# Initialize the parameters of the Bayesian convolutional neural network.
theta = {
    model_W_conv_1: 0.1 * np.random.randn(n_particles, 8, 8, 1, 16),
    model_W_conv_2: 0.1 * np.random.randn(n_particles, 4, 4, 16, 32),
    model_b_conv_1: np.zeros((n_particles, 16)) + 0.1,
    model_b_conv_2: np.zeros((n_particles, 32)) + 0.1,
    model_W_1: 0.1 * np.random.randn(n_particles, 4*4*32, 1024),
    model_W_2: 0.1 * np.random.randn(n_particles, 1024, 10),
    model_b_1: np.zeros((n_particles, 1024)) + 0.1,
    model_b_2: np.zeros((n_particles, 10)) + 0.1
}

# Sample from the posterior using Stein variational gradient descent.
gd = AdamGradientDescent(learning_rate=1e-2, decay=0.999)
sampler = SteinSampler(n_particles, log_p, gd)
# Perform learning iterations.
current_iter = 0
while True:
    if current_iter % n_prog == 0:
        acc = evaluation(sampler, {
            model_X: mnist.test.images, model_y: mnist.test.labels
        })
        print("Iteration: {}\t\t{}".format(current_iter, acc))
    batch = mnist.train.next_batch(n_batch)
    sampler.train_on_batch({model_X: batch[0], model_y: batch[1]})
    current_iter += 1



