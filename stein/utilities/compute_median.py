import tensorflow as tf


def compute_median(D):
    """Computes the median value of the input."""
    # Get the shape of the values for which we'll compute the median.
    V = tf.reshape(D, [-1])
    dim = V.shape[0].value
    m = dim // 2 + 1
    # The formula for computing the median is dependent on whether or not we
    # have an even or odd number of points.
    if dim % 2 == 0:
        median = tf.reduce_mean(tf.nn.top_k(V, m).values[m - 2:])
    else:
        median = tf.nn.top_k(V, m).values[m - 1]
    return median
