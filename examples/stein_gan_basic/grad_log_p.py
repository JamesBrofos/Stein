import numpy as np


def grad_log_p(x, params):
    """Compute the gradient with respect to the input to the log-target
    distribution with fixed parameters. In this case, the target distribution
    is a mixture of two Gaussians and the parameters are the means, with known
    scale parameters.

    Here is the Wolfram Alpha code to reproduce this computation:

        d/dx log(
            1/2*1/sqrt(2*pi) * exp(-(x - theta)^2 / (2 * phi))
        )
    """
    mu, sigma = params[0], params[1]
    return -(x - mu) / sigma
