import tensorflow as tf
#tensors are multidimensional matrices

'''this is a 10 neuron, 1 layer network here'''

#import images to train and test
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


#None will become the batch size
#28x28 is the image dimensions
#1 means grayscale images / number of values per pixel
#placeholder is where we will be inputting our data

X = tf.placeholder(tf.float32, [None, 28, 28, 1])

#variables are the things we want tensorflow to determine for us
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10])

init = tf.initialize_all_variables()

#model  -reshape is flattening the images (the -1 means there is only 1 solution)  the model gives us predictions (activation function)
Y = tf.nn.softmax(tf.matmul(tf.reshape(X, [-1, 784]), W) + b)

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

sess = tf.Session()
sess.run(init)


#loop through 60k training images in batches of 100 to process 100k images total
for i in range(1000):
#load batch of images and correct answers in groups of 100
    batch_X, batch_Y = mnist.train.next_batch(100)
    train_data = {X: batch_X, Y_: batch_Y}

    #train on that data using the data from our image batch of 100
    sess.run(train_step, feed_dict=train_data)

    #see how successful we are on our training data
    a, c = sess.run([accuracy, cross_entropy], feed=train_data)


    #see how successful we are on our testing data (run this only every so often, not on every iteration of the training loop)
    test_data = {X: mnist.test.images, Y_: mnist.test.labels}
    a,c = sess.run([accuracy, cross_entropy], feed=test_data)


