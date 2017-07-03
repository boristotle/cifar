#classification (MNIST)
#https://www.youtube.com/watch?v=AhC6r4cwtq0&list=PLXO45tsB95cJHXaDKpbwr5fC_CCYylw1f&index=16


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def add_layer(inputs, input_size, output_size, activation_function=None):
    #add one more layer and return output layer
    Weights = tf.Variable(tf.random_normal([input_size, output_size]))
    biases = tf.Variable(tf.zeros([1, output_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


def compute_accuracy(v_xs, v_ys):
    global predicted_labels
    y_pre = sess.run(predicted_labels, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result

#define placeholder for inputs

#input placeholders
xs = tf.placeholder(tf.float32, [None, 28*28])
ys = tf.placeholder(tf.float32, [None, 10])



#add output layer
predicted_labels = add_layer(xs, 28*28, 10, activation_function=tf.nn.softmax)


#calculate error between prediction and real data
cross_entropy = -tf.reduce_sum(ys * tf.log(predicted_labels))

optimizer = tf.train.GradientDescentOptimizer(0.003)
train_step = optimizer.minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(10000):
    batch_features, batch_labels = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_features, ys: batch_labels})
    if (i % 50 == 0):
        print(compute_accuracy(mnist.test.images, mnist.test.labels))