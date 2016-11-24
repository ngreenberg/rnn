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

    vocab = [w for w in vocab.keys() if vocab[w] >= 10]
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


def generate_matrix(x, y, tags):

    x = pd.DataFrame(x)
    x = x.fillna(0).astype(int).values

    num_classes = len(tags)

    y = [np.eye(num_classes)[s] for s in y]
    y = [s.tolist() for s in y]
    y = pd.DataFrame(y)
    y = fill_one_hot(y, num_classes)

    return x, y
