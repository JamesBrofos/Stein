import numpy as np
import tensorflow as tf
from tensorflow.contrib.distributions import Normal


# For reproducibility.
if True:
    np.random.seed(0)

# Sample from the distribution.
n_samples = 10000
data_X = np.zeros((n_samples))
for i in range(n_samples):
    if np.random.uniform() < 1. / 3:
        data_X[i] = np.random.normal(-2., 1.)
    else:
        data_X[i] = np.random.normal(2., 1.)

# TensorFlow model.
with tf.variable_scope("model"):
    model_X = tf.Variable(tf.zeros([]))
    p = (
        1. / 3 * Normal(-2., 1.).prob(model_X) +
        2. / 3 * Normal(2., 1.).prob(model_X)
    )
    log_p = tf.log(p)

# Number of particles.
n_particles = 50
# Initialize the particles to a bad prior.
theta = {
    model_X: np.random.normal(-10., 1., size=(n_particles, ))
}
