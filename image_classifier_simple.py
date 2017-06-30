import tensorflow as tf
import pickle
import numpy as np

import matplotlib.pyplot as plt

def one_hot_encode(x):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
    """
    nx = np.max(x) + 1
    return np.eye(nx)[x]


meta_file = open('cifar-10-batches-py/batches.meta', "rb")
dict = pickle.load(meta_file)

label_names = dict["label_names"]
#print('dict', dict["label_names"])



img_data_file = open('cifar-10-batches-py/data_batch_1', "rb")
img_dict = pickle.load(img_data_file, encoding="latin1")
#print('img_data', img_dict['data'])
#print('img_labels', img_dict['labels'])
#print('one_hot_encoded_labels', one_hot_encode(img_dict['labels'][0]))

#train_features = img_dict['data'].reshape((len(img_dict['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
train_labels = one_hot_encode(img_dict['labels'])
train_features = img_dict['data']

#print('train_features', len(train_features[0]))
#print('train_labels', train_labels)



test_data_file = open('cifar-10-batches-py/test_batch', "rb")
test_dict = pickle.load(test_data_file, encoding="latin1")

test_features = test_dict['data']
test_labels = test_dict['labels']

X = tf.placeholder(tf.float32, [None, 3072], 'X')
W = tf.Variable(tf.zeros([32*32*3, 10]))
b = tf.Variable(tf.zeros([10]))


Y = tf.nn.softmax(tf.matmul(tf.reshape(X, [-1, 32*32*3]), W) + b)

#placeholder for correct answers
Y_ = tf.placeholder(tf.float32, [None, 10], 'Y_')


cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y))

is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

optimizer = tf.train.GradientDescentOptimizer(0.0003)
train_step = optimizer.minimize(cross_entropy)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
print('train_features[0]', train_features[0])
print('train_labels[0]', train_labels[0])
'''
train_data = {X: train_features[0], Y_: train_labels[0]}
sess.run(train_step, feed_dict=train_data)
a, c = sess.run([accuracy, cross_entropy], feed_dict=train_data)
print('accuracy_training', a)

'''


for i in range(10):
#load batch of images and correct answers

    train_data = {X: train_features, Y_: train_labels}

    #train on that data using the data from our image batch of 100
    sess.run(train_step, feed_dict=train_data)

    #see how successful we are on our training data
    a, c = sess.run([accuracy, cross_entropy], feed_dict=train_data)
    print('accuracy_training', a)
    print('cross_entropy', c)
    print('train_labels[i]', train_labels[i])


    #see how successful we are on our testing data (run this only every so often, not on every iteration of the training loop)
    #test_data = {X: mnist.test.images, Y_: mnist.test.labels}
    #a,c = sess.run([accuracy, cross_entropy], feed_dict=test_data)
    #print('accuracy_test', a)

