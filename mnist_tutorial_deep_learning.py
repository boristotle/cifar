import tensorflow as tf
#tensors are multidimensional matrices

'''each neuron does a weighted sum of all the pixels in each image, then a weighted sum of all the outputs for each layer'''
'''this is a 10 neuron, 5 layer network here
use softmax for our output layer
use RELU function for our intermediate layers
need one weights matrix and one bias vector per layer


large peaks and valleys in charts means our learning rate is too fast so we need a learning rate decay as we go forward

overfitting if loss goes down then starts to go back up on chart
overfitting results from too many neurons or not enough data or bad network
(dropout in training can be used to prevent overfitting, will not use dropout in testing)
dropout is applied after each layer except after the output layer

pkeep = tf.placeholder(tf.float32)
Yf = tf.nn.relu(tf.matmul(X, W) + B)
Y = tf.nn.dropout(Yf, pkeep)
'''

#import images to train and test
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

X = tf.placeholder(tf.float32, [None, 28, 28, 1])

#4 layers with x number of neurons with the last, output layer having 10 neurons
K = 200
L = 100
M = 60
N = 30

#truncated normal initializes the weights as small random values as good practice
W1 = tf.Variable(tf.truncated_normal([28*28, K], stddev=0.1))
B1 = tf.Variable(tf.zeros([K]))

W2 = tf.Variable(tf.truncated_normal([K,L], stddev=0.01))
B2 = tf.Variable(tf.zeros([L]))

W3 = tf.Variable(tf.truncated_normal([L,M], stddev=0.1))
B3 = tf.Variable(tf.zeros([M]))

W4 = tf.Variable(tf.truncated_normal([M,N], stddev=0.1))
B4 = tf.Variable(tf.zeros([N]))

W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
B5 = tf.Variable(tf.zeros([10]))


X = tf.reshape(X, [-1, 28*28])

Y1 = tf.nn.relu(tf.matmul(X, W1) + B1)
Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2)
Y3 = tf.nn.relu(tf.matmul(Y2, W3) + B3)
Y4 = tf.nn.relu(tf.matmul(Y3, W4) + B4)
Y = tf.nn.softmax(tf.matmul(Y4, W5) + B5)


#placeholder for correct answers
Y_ = tf.placeholder(tf.float32, [None, 10])

#loss function  (comparing our predictions to the known labels)
cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y))

# % of correct answers found in batch
is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))







#TRAINING FUNCTIONS (pick an optimizer and minimize loss/cross-entropy)
optimizer = tf.train.GradientDescentOptimizer(0.003)
train_step = optimizer.minimize(cross_entropy)

#RUN TRAINING LOOP
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


#loop through 100k images in batches of 100
for i in range(1000):
#load batch of images and correct answers in groups of 100
    batch_X, batch_Y = mnist.train.next_batch(100)
    train_data = {X: batch_X, Y_: batch_Y}

    #train on that data using the data from our image batch of 100
    sess.run(train_step, feed_dict=train_data)

    #see how successful we are on our training data
    a, c = sess.run([accuracy, cross_entropy], feed_dict=train_data)


    #see how successful we are on our testing data (run this only every so often, not on every iteration of the training loop)
    test_data = {X: mnist.test.images, Y_: mnist.test.labels}
    a,c = sess.run([accuracy, cross_entropy], feed_dict=test_data)
    if (i % 50 == 0):
        print('accuracy', a)






