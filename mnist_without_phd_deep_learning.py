import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

X = tf.placeholder(tf.float32, [None, 784])



K = 200
L = 100
M = 60
N = 30
output_neurons = 10

W1 = tf.Variable(tf.truncated_normal([784, K], stddev=0.1))
B1 = tf.Variable(tf.zeros([K]))

W2 = tf.Variable(tf.truncated_normal([K, L], stddev=0.1))
B2 = tf.Variable(tf.zeros([L]))

W3 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
B3 = tf.Variable(tf.zeros([M]))

W4 = tf.Variable(tf.truncated_normal([M,N], stddev=0.1))
B4 = tf.Variable(tf.zeros([N]))

W5 = tf.Variable(tf.truncated_normal([N, output_neurons], stddev=0.1))
B5 = tf.Variable(tf.zeros([10]))


X = tf.reshape(X, [-1, 784])

Y1 = tf.nn.relu(tf.matmul(X, W1) + B1)
Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2)
Y3 = tf.nn.relu(tf.matmul(Y2, W3) + B3)
Y4 = tf.nn.relu(tf.matmul(Y3, W4) + B4)


#model
Y = tf.nn.softmax(tf.matmul(Y4, W5) + B5)


#placeholder for correct answers
Y_ = tf.placeholder(tf.float32, [None, 10])



#loss function
cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y))

#% of correct answers found in batch
is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

optimizer = tf.train.GradientDescentOptimizer(0.003)
train_step = optimizer.minimize(cross_entropy)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

#loop through 60k training images in batches of 100 to process 100k images total
for i in range(10000):
#load batch of images and correct answers in groups of 100
    batch_X, batch_Y = mnist.train.next_batch(100)
    train_data = {X: batch_X, Y_: batch_Y}

    #train on that data using the data from our image batch of 100
    sess.run(train_step, feed_dict=train_data)

    #see how successful we are on our training data
    a, c = sess.run([accuracy, cross_entropy], feed_dict=train_data)
    print('accuracy_training', a)


    #see how successful we are on our testing data (run this only every so often, not on every iteration of the training loop)
    test_data = {X: mnist.test.images, Y_: mnist.test.labels}
    a,c = sess.run([accuracy, cross_entropy], feed_dict=test_data)
    print('accuracy_test', a)

