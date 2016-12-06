from __future__ import division

from collections import defaultdict
import numpy as np


def _pad_to_max_length(data):
    max_length = max([len(l) for l in data])
    shape = (len(data), max_length) + np.shape(data[0])[1:]

    out = np.zeros(shape)
    for i, l in enumerate(data):
        out[i, :len(l)] = l
    return out


def _index_data(line, indices):
    indexed_data = []
    for w in line:
        if w not in indices.keys():
            indexed_data.append(indices['UNK'])
        else:
            indexed_data.append(indices[w])
    return indexed_data


class DataProcessor(object):

    def __init__(self):
        self.data = dict()

        self.vocab = None
        self.inv_vocab = None

        self.tags = None
        self.inv_tags = None

    def index_words(self, line):
        return _index_data(line, self.vocab)

    def inverse_index_words(self, line):
        return _index_data(line, self.inv_vocab)

    def index_tags(self, line):
        return _index_data(line, self.tags)

    def inverse_index_tags(self, line):
        return _index_data(line, self.inv_tags)

    def read_file(self, filepath, key):
        f = open(filepath, 'r')

        x, y = [], []

        if not self.vocab:
            vocab_list = defaultdict(int)
            tags_list = set()

            build_vocab = True
        else:
            build_vocab = False

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

                if build_vocab:
                    vocab_list[word] += 1
                    tags_list.add(tag)

        f.close()

        if build_vocab:
            # remove words with a word count of 1
            vocab_list = [w for w in vocab_list.keys() if vocab_list[w] >= 1]
            vocab_list.append('UNK')

            self.vocab = {w: i + 1 for i, w in enumerate(vocab_list)}
            self.inv_vocab = {i: w for w, i in self.vocab.iteritems()}

            self.tags = {t: i for i, t in enumerate(tags_list)}
            self.inv_tags = {i: t for t, i in self.tags.iteritems()}

        x = [self.index_words(l) for l in x if l != []]
        y = [self.index_tags(l) for l in y if l != []]

        self.data[key] = (x, y)

    def _convert_to_matrix(self, x, y):
        y = [np.eye(len(self.tags))[l] for l in y]

        return (_pad_to_max_length(x), _pad_to_max_length(y))

    def next_batch(self, key, size):
        batch = np.random.choice(len(self.data[key][0]), size)

        x = [self.data[key][0][i] for i in batch]
        y = [self.data[key][1][i] for i in batch]

        return self._convert_to_matrix(x, y)

    def get_data(self, key):
        x = self.data[key][0]
        y = self.data[key][1]

        return self._convert_to_matrix(x, y)
