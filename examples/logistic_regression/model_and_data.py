import tensorflow as tf
from scipy import io
from sklearn.model_selection import train_test_split
from tensorflow.contrib.distributions import Normal, Gamma


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


