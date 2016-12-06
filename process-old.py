from __future__ import division

from collections import defaultdict
import math

import numpy as np
import pandas as pd


def index_data(words, indices):
    indexed_data = []
    for w in words:
        if w not in indices.keys():
            indexed_data.append(indices['UNK'])
        else:
            indexed_data.append(indices[w])
    return indexed_data


def process_main(filepath):
    f = open(filepath, 'r')

    x, y = [], []
    vocab, tags = defaultdict(int), set()

    for line in f:
        if line.startswith('-DOCSTART-'):
            continue
        elif line == '\n':
            x.append([]), y.append([])
        else:
            split = line.split()
            word, tag = split[0], split[1]

            x[-1].append(word)
            y[-1].append(tag)

            vocab[word] += 1
            tags.add(tag)

    f.close()

    x = [l for l in x if l != []]
    y = [l for l in y if l != []]

    vocab = [w for w in vocab.keys() if vocab[w] >= 1]
    vocab.append('UNK')

    vocab = {w: i + 1 for i, w in enumerate(vocab)}
    tags = {t: i for i, t in enumerate(tags)}

    x = [index_data(s, vocab) for s in x]
    y = [index_data(s, tags) for s in y]

    return x, y, vocab, tags


def process_additional(filepath, vocab, tags):
    f = open(filepath, 'r')

    x, y = [], []

    for line in f:
        if line.startswith('-DOCSTART-'):
            continue
        elif line == '\n':
            x.append([]), y.append([])
        else:
            split = line.split()
            word, tag = split[0], split[1]

            x[-1].append(word)
            y[-1].append(tag)

    f.close()

    x = [l for l in x if l != []]
    y = [l for l in y if l != []]

    x = [index_data(s, vocab) for s in x]
    y = [index_data(s, tags) for s in y]

    return x, y


def fill_one_hot(batch, length):
    output = []
    for sequence in batch.values:
        output.append([[0.0] * length if n is None else n for n in sequence])
    return np.array(output)


def generate_batches(x, y, tags, batch_size):
    num_batches = math.ceil(len(x) / batch_size)
    x_batches = np.array_split(x, num_batches)
    y_batches = np.array_split(y, num_batches)

    x_batches = [pd.DataFrame(b.tolist()) for b in x_batches]
    x_batches = [b.fillna(0).astype(int).values for b in x_batches]

    num_classes = len(tags)

    y_batches = [[np.eye(num_classes)[s] for s in b] for b in y_batches]
    y_batches = [[s.tolist() for s in b] for b in y_batches]
    y_batches = [pd.DataFrame(b) for b in y_batches]
    y_batches = [fill_one_hot(b, num_classes) for b in y_batches]

    return x_batches, y_batches


def generate_matrix(x, y, tags):
    a, b = generate_batches(x, y, tags, len(x))
    return a[0], b[0]
