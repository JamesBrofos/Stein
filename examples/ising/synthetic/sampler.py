import numpy as np
from utilities import enumerate_binary, construct_pairwise_interactions


# For reproducibility.
np.random.seed(0)

# Number of nodes and weights in the fully connected graph.
n_nodes = 10
n_weights = (n_nodes * (n_nodes - 1)) // 2
# Sample weights from a standard normal distribution and the biases from a
# uniform distribution.
W = np.random.normal(size=(n_weights, ))
b = np.random.uniform(-0.1, 0.1, size=(n_nodes, ))

# Create all binary combinations.
B = enumerate_binary(n_nodes)
P = construct_pairwise_interactions(B)
# Compute partition function and probability vector for each sample.
F = np.exp(P.dot(W) + B.dot(b))
Z = F.sum()
probs = F / Z
# Compute the cumulative probability vector which will provide our cutoffs for
# exact sampling.
cum_probs = probs.cumsum()
# Extract the number of fully enumerated binary combinations.
n_binary = 2 ** n_nodes


# State the number of samples to draw.
n_samples = 1000
# Perform sampling.
indices = np.random.choice(n_binary, size=(n_samples, ), p=probs)
samples = B[indices]

# Save variables to file.
np.savetxt("./data/weights.csv", W, delimiter=",")
np.savetxt("./data/biases.csv", b, delimiter=",")
np.savetxt("./data/samples.csv", samples, delimiter=",")

