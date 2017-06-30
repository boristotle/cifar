#https://www.youtube.com/watch?v=Kd7gDHY_OUU&list=PLXO45tsB95cJHXaDKpbwr5fC_CCYylw1f&index=9
#activation function (happens on the weights and biases in every layer)

#https://www.youtube.com/watch?v=Vu_lIJ_Yexk&list=PLXO45tsB95cJHXaDKpbwr5fC_CCYylw1f&index=10
#add layer

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs, input_size, output_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([input_size, output_size])) #capital W in weights bc it is 2 dimensional value
    biases = tf.Variable(tf.zeros([1, output_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if (activation_function is None):
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

#make up some real data
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
#print('x_data', x_data)
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise


#plot data
#plt.scatter(x_data, y_data)
#plt.show()


#define placeholder for inputs to network
# none represents the number of samples, 1 is how many features
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

#add hidden layer
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)


#add output layer
prediction = add_layer(l1, 10, 1, activation_function=None)


#calculate error between prediction and real data
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train_step = optimizer.minimize(loss)

#init variables
init = tf.global_variables_initializer()

#run training in session
sess = tf.Session()
sess.run(init)

for i in range(1000):
    #training
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if (i % 50 ==0):
        print('loss', sess.run(loss, feed_dict={xs: x_data, ys: y_data}))