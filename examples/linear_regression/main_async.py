import numpy as np
import tensorflow as tf
import os
from tensorflow.contrib.distributions import Normal
from mpi4py import MPI
from stein.gradient_descent import AdamGradientDescent
from stein.kernels import SquaredExponentialKernel
from stein.utilities import (
    convert_array_to_dictionary,
    convert_dictionary_to_array
)


# For reproducibility.
np.random.seed(0)

# TensorFlow logging level.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# Create the distribution communications controller.
comm = MPI.COMM_WORLD
# Number of features and number of observations.
n_samples, n_feats = 10000, 1
data_X = np.random.normal(size=(n_samples, n_feats))
data_w = np.random.normal(scale=3., size=(n_feats, 1))
data_y = np.random.normal(data_X.dot(data_w), 0.1)

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


# Number of learning iterations.
n_iters = 1000
n_prog = n_iters // 10

# Interpret the number of processes as the number of particles to sample (this
# is done for simplicity). We subtract one because we have a single master
# process.
n_particles = comm.size - 1

# Do TensorFlow setup.
sess = tf.Session()
model_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "model")
grad_log_p = tf.gradients(log_p, model_vars)

# Variables for master process.
if comm.rank == 0:
    # We need to initialize a set of global parameters.
    theta = {
        v: np.random.normal(size=[n_particles] + v.get_shape().as_list())
        for v in model_vars
    }
    # Kernel for Stein variational gradient descent.
    kernel = SquaredExponentialKernel()
    # Gradient descent method for Stein variational gradient descent.
    gd = AdamGradientDescent(learning_rate=1e-1)


# Perform Stein variational gradient descent iterations.
for i in range(n_iters):
    if comm.rank == 0:
        # This is the master process. The master process is responsible for
        # coordinating the gradient computations and combining them to update
        # the optimal perturbation direction.
        for i in range(n_particles):
            param_dict = {v.name: theta[v][i] for v in model_vars}
            comm.send(param_dict, dest=i+1)

        # Initialize a dictionary to store the gradient with respect to each
        # constituent parameter of the particle.
        grads = {
            v: np.zeros([n_particles] + v.get_shape().as_list())
            for v in model_vars
        }
        for i in range(n_particles):
            grad_dict = comm.recv(source=i+1)
            for v, g in grad_dict.items():
                var = next((x for x in model_vars if x.name == v), None)
                if var is None:
                    raise ValueError("Could not find variable.")
                grads[var][i] = g

        # Convert both the particle dictionary and the gradient dictionary into
        # vector representations.
        theta_array, access_indices = convert_dictionary_to_array(theta)
        grads_array, _ = convert_dictionary_to_array(grads)
        # Extract the number of particles and number of parameters.
        n_params = grads_array.shape[1]
        # Compute the kernel matrices and gradient with respect to the
        # particles.
        K, dK = kernel.kernel_and_grad(theta_array)
        phi = (K.dot(grads_array) + dK) / n_particles
        # Normalize the gradient have be norm no larger than the desired amount.
        phi *= 10. / max(10., np.linalg.norm(phi))
        theta_array += gd.update(phi)
        theta = convert_array_to_dictionary(theta_array, access_indices)

    else:
        # This is the worker process. It is responsible for computing the
        # gradient of the posterior log-likelihood with respect to the model
        # parameters.
        data = comm.recv(source=0)
        theta_feed = {}
        for v in data:
            var = next((x for x in model_vars if x.name == v), None)
            if var is None:
                raise ValueError("Could not find variable.")
            theta_feed[var] = data[v]
        # Save space by clearing out the transmitted dictionary.
        del data
        # Update the variable feed dictionary with the data placeholders.
        theta_feed.update({model_X: data_X, model_y: data_y})

        # Compute the gradient of the log-posterior with respect to the model
        # parameters.
        grad = sess.run(grad_log_p, theta_feed)
        grad_dict = {v.name: g for v, g in zip(model_vars, grad)}
        # Send the gradient back to the master node.
        comm.send(grad_dict, dest=0)

if comm.rank == 0:
    print(data_w)
    print(theta)
