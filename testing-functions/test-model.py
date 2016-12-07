from __future__ import division

from tqdm import tqdm, trange

import testfunction

import tensorflow as tf
from bilstm import BiLSTM


data = tf.placeholder(tf.int32, [None, None])
target = tf.placeholder(tf.float32, [None, None, 6])

model = BiLSTM(data, target, 6, embedding_size=100,
               lstm_size=100, learning_rate=0.01)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

epochs = 20000
for i in trange(epochs):
    if i % 100 == 0:
        x, y = testfunction.get_training_data()
        training_error = sess.run(model.error, feed_dict={data: x, target: y})

        x, y = testfunction.get_testing_data()
        testing_error = sess.run(model.error, feed_dict={data: x, target: y})

        tqdm.write("training error: " + str(training_error) +
                   ", testing error: " + str(testing_error))

    x, y = testfunction.next_batch(100)
    sess.run(model.optimize, feed_dict={data: x, target: y})
