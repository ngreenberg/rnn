from __future__ import division

from tqdm import tqdm, trange

from processtest import DataProcessor

import tensorflow as tf
from bilstm import BiLSTM


################
# Process data #
################

print "Processing data..."

batch_size = 100

dp = DataProcessor()
dp.read_file('./data/eng.train', 'train')
dp.read_file('./data/eng.testa', 'test')


###############
# Build model #
###############

num_classes = len(dp.tags)
vocab_size = len(dp.vocab) + 1

data = tf.placeholder(tf.int32, [None, None])
target = tf.placeholder(tf.float32, [None, None, num_classes])

model = BiLSTM(data, target, vocab_size, embedding_size=100,
               lstm_size=100, learning_rate=0.01)


###############
# Train model #
###############

print "Training model..."

sess = tf.Session()
sess.run(tf.initialize_all_variables())

epochs = 100000
for i in trange(epochs):
    if i % 100 == 0:
        x, y = dp.get_data('test')
        tqdm.write(str(sess.run(model.error,
                                feed_dict={data: x, target: y})))

    x, y = dp.next_batch('train', batch_size)
    sess.run(model.optimize, feed_dict={data: x, target: y})


##############
# Test model #
##############

print
x, y = dp.get_data('test')
print sess.run(model.error, feed_dict={data: x, target: y})
