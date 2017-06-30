#https://www.youtube.com/watch?v=PFijwks2K6o&list=PLXO45tsB95cJHXaDKpbwr5fC_CCYylw1f&index=5

import tensorflow as tf
import numpy as np

#create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

#create tensorflow structure start
Weights = tf.Variable(tf.truncated_normal([1], stddev=0.1))
biases = tf.Variable(tf.zeros([1]))

#predicted y (model)
y = Weights * x_data + biases

#loss is error between prediction and actual values
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)

train_step = optimizer.minimize(loss)

init = tf.initialize_all_variables() #initializes the variables in tensorflow


sess = tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train_step)
    if step % 20 == 0:
        print('step', step, 'weight', sess.run(Weights), 'biases', sess.run(biases))


