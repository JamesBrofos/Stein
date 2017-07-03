import numpy as np


def convert_dictionary_to_array(dictionary):
    """"""
    n_particles = next(iter(dictionary.values())).shape[0]
    n_params = sum([value[0].size for value in dictionary.values()])
    array = np.zeros((n_particles, n_params))
    access_indices = {}
    index = 0
    for v in dictionary:
        value = dictionary[v]
        dim = int(np.prod(value.shape[1:]))
        value = np.reshape(value, (n_particles, dim))
        array[:, index:index+dim] = value
        access_indices[v] = (index, index+dim)
        index += dim

    return array, access_indices

def convert_array_to_dictionary(array, access_indices):
    """"""
    dictionary = {}
    n_particles = array.shape[0]
    for v, indices in access_indices.items():
        dictionary[v] = np.reshape(
            array[:, indices[0]:indices[1]],
            [n_particles] + v.get_shape().as_list()
        )

    return dictionary
