from __future__ import division

import numpy as np

sequence_length_range = xrange(10, 16)
value_range = xrange(1, 6)

dataset_size = 5000


def classify_sequence(sequence):
    last_index = len(sequence) - 1

    labels = [.5 * sequence[i - 1] + sequence[i] + 2 * sequence[i + 1]
              for i in xrange(1, last_index)]

    first_label = 1 + sequence[0] + 2 * sequence[1]
    last_label = .5 * sequence[last_index - 1] + sequence[last_index] + 5
    labels = [first_label] + labels + [last_label]

    labels = [int(n / 3) for n in labels]

    return labels


def generate_sequence():
    return np.random.choice(value_range,
                            np.random.choice(sequence_length_range))


def _pad_to_max_length(data):
    max_length = max([len(l) for l in data])
    shape = (len(data), max_length) + np.shape(data[0])[1:]

    out = np.zeros(shape)
    for i, l in enumerate(data):
        out[i, :len(l)] = l
    return out


def generate_data():
    x = [generate_sequence() for _ in xrange(dataset_size)]
    y = [classify_sequence(sequence) for sequence in x]

    split_index = int(dataset_size * .8)

    training = (x[:split_index], y[:split_index])
    testing = (x[split_index:], y[split_index:])

    return training, testing


training, testing = generate_data()


def convert_to_matrix(x, y):
    y = [np.eye(6)[sequence] for sequence in y]

    return _pad_to_max_length(x), _pad_to_max_length(y)


def next_batch(size):
    batch = np.random.choice(len(training[0]), size)

    x = [training[0][i] for i in batch]
    y = [training[1][i] for i in batch]

    return convert_to_matrix(x, y)


def get_training_data():
    return convert_to_matrix(training[0], training[1])


def get_testing_data():
    return convert_to_matrix(testing[0], testing[1])
