import numpy as np


def convert_dictionary_to_array(dictionary):
    """This method takes a dictionary mapping TensorFlow variables to a matrix
    where each row corresponds to a particle and each column is a value of a
    parameter for that variable. This method returns a tuple consisting of a
    matrix representation of all of the variables in the dictionary and their
    corresponding values, as well as another dictionary that provides index
    information on how to extract the parameters corresponding to a particular
    variable from the matrix.

    Parameters:
        dictionary (dict): A dictionary mapping TensorFlow variables to matrices
            where each row is a particle and each column is a parameter for that
            variable.

    Returns:
        Tuple: The first element of the tuple is the matrix representation of
            the dictionary. Essentially this is the concatenation of the
            matrices that were the values in the input dictionary, but they are
            reshaped to have dimensions equal to the number of particles by the
            number of parameters in that variable. The second output is a
            dictionary mapping TensorFlow variables to 2-tuples, which provide
            the start and end index into the output matrix for each variable.
    """
    # Number of particles and number of parameters. Notice that by construction
    # the first element of the size of any value in the input dictionary is the
    # number of particles.
    n_particles = next(iter(dictionary.values())).shape[0]
    n_params = sum([value[0].size for value in dictionary.values()])
    array = np.zeros((n_particles, n_params))
    # Initialize an array to store the mapping of indices into the output matrix
    # to be used for accessing a given variable.
    access_indices = {}
    index = 0
    # Here we sort the TensorFlow variables that represent the keys to the
    # dictionary. This ensures consistency in the way that the gradients and
    # parameters are stored.
    variables = [v for v in dictionary]
    variables.sort(key=lambda x: x.name)

    # Iterate over each of the variables.
    for v in variables:
        # Extract the corresponding value from the dictionary.
        value = dictionary[v]
        # Compute the number of parameters in each variable. This is the product
        # of the individual dimensions of that variable (excluding the first
        # which is just the number of particles).
        dim = int(np.prod(value.shape[1:]))
        value = np.reshape(value, (n_particles, dim))
        # Store the reshaped matrix in the output matrix and record the output
        # indices for later use.
        array[:, index:index+dim] = value
        access_indices[v] = (index, index+dim)
        index += dim

    return array, access_indices


def convert_array_to_dictionary(array, access_indices):
    """This method takes a matrix representation of the variables in a model and
    outputs a dictionary which maps variables to the the values assigned to each
    parameter in that variable for each particle.

    Parameters:
        array (numpy array): A matrix representation of the values of every
            variable defined in the TensorFlow model. The dimensions of this
            matrix are the number of particles by the number of parameters.
        access_indices (dict): A dictionary mapping variables to their
            corresponding integer indices in the `array` variable. This allows
            us to extract the corresponding submatrix and store it in the output
            dictionary.

    Returns:
        Dict: A dictionary mapping TensorFlow variables to matrices where each
            row is a particle and each column is a parameter for that variable.
    """
    # Initialize the dictionary and compute the number of particles.
    dictionary = {}
    n_particles = array.shape[0]
    # Iterate over each of the variables and the corresponding access indices
    # into the matrix representation of the model variables.
    for v, indices in access_indices.items():
        # Make to reshape the extracted array into the shape originally defined
        # by the variable.
        dictionary[v] = np.reshape(
            array[:, indices[0]:indices[1]],
            [n_particles] + v.get_shape().as_list()
        )

    return dictionary
