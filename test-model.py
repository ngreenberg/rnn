from __future__ import division

import process
import numpy as np

import tensorflow as tf
from bilstm import BiLSTM


################
# Process data #
################

print "Processing data..."

x, y, vocab, tags = process.process_main('./data/eng.train')
x, y = process.generate_matrix(x, y, tags)


###############
# Build model #
###############

max_length = x.shape[1]
num_classes = len(tags)

data = tf.placeholder(tf.int32, [None, None])
target = tf.placeholder(tf.float32, [None, None, num_classes])

model = BiLSTM(data, target, len(vocab) + 1, embedding_size=20, lstm_size=20)


###############
# Train model #
###############

print "Training model..."

batch_size = 50
num_batches = len(x) // batch_size
x_batches = np.array_split(x, num_batches)
y_batches = np.array_split(y, num_batches)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

epochs = 10
for i in range(epochs):
    if i % 1 == 0:
        print sess.run(model.cost,
                       feed_dict={data: x, target: y}),
        print sess.run(model.error,
                       feed_dict={data: x, target: y})
    for x, y in zip(x_batches, y_batches):
        sess.run(model.optimize, feed_dict={data: x, target: y})


##############
# Test model #
##############

print
# print sess.run(model.error, feed_dict={data: test_input, target:
# test_output})
