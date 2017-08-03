import numpy as np


# Number of nodes and weights in the fully connected graph.
n_nodes = 6
n_weights = (n_nodes * (n_nodes - 1)) // 2
# Sample weights from a standard normal distribution and the biases from a
# uniform distribution.
W = np.random.normal(size=(n_weights, ))
b = np.random.uniform(-0.1, 0.1, size=(n_nodes, ))


def unnormalized_log_probability(s):
    # First we need to compute the pairwise interactions.
    p = np.zeros((n_weights, ))
    index = 0
    for i in range(1, n_nodes):
        for j in range(i, n_nodes):
            p[index] = s[i] * s[j]
            index += 1
    # Then we compute the unnormalized probability by simply computing the
    # linear combinations of the bias terms and interaction terms.
    return W.dot(p) + b.dot(s)

def sample(n_samples, n_burn):
    # Create a matrix to store the samples. This is initialized to be larger
    # than necessary since the burn-in samples will be included initially and
    # then filtered out.
    X = np.zeros((n_burn + n_samples, n_nodes))
    # Create a counter to store the effective number of samples drawn so far.
    m = 0
    # Initialize a current state at random for a Bernoulli-1/2 distribution.
    s = (np.random.uniform(size=(n_nodes, )) < 0.5).astype(np.float64)

    # Iterate until we have drawn all the samples we need.
    while m < n_burn + n_samples:
        # Compute the unnormalized probability for the current state.
        prob_s = unnormalized_log_probability(s)
        # Choose a random index into the state vector and flip the corresponding
        # bit. This represents a proposal for sampling from the Ising model.
        index = np.random.randint(n_nodes)
        sp = s.copy()
        sp[index] = 1. - sp[index]
        # Compute the unnormalized probability of the proposal state.
        prob_sp = unnormalized_log_probability(sp)
        # Accept or reject the proposal according to its likelihood.
        transition = min(0., prob_sp - prob_s)
        if np.log(np.random.uniform()) < transition:
            # If the sample is accepted then record it in the matrix of samples
            # and increment the effective samples counter.
            X[m] = sp
            m += 1
            s = sp

    # Take only the samples after the burn-in period.
    return X[n_burn:]

# Number of samples to collect and number of burn-in iterations to perform.
n_samples = 10000
n_burn = 9000
# Sample from the binary Ising model.
X = sample(n_samples, n_burn)

# Save variables to file.
np.savetxt("./data/weights.csv", W, delimiter=",")
np.savetxt("./data/biases.csv", b, delimiter=",")
np.savetxt("./data/samples.csv", X, delimiter=",")
