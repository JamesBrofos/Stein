import numpy as np


# Number of observations and number of covariates.
n, k = 1000, 1
# Create linear system.
data_X = np.random.normal(size=(n, k))
data_w = np.random.normal(size=(k, 1)) * 5
data_y = np.random.normal(data_X.dot(data_w), 0.3)

# Save data to file.
np.savetxt("data_X.csv", data_X, delimiter=",")
np.savetxt("data_w.csv", data_w, delimiter=",")
np.savetxt("data_y.csv", data_y, delimiter=",")
