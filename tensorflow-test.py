import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/mnist_data/", one_hot=True)


'''neural network with 1 layer of 10 softmax neurons

(input data, flattened pixels)  X [batch, 784]
                                W [784,10]
                                Y [batch, 10]

The model is:

Y = softmax ( X * W + b)
            X: matrix for 100 grayscale images 28x28 pixels, flattened (there are 100 images in the mini batch)
            W: weight matrix with 784 lines and 10 columns
            b: bias vector with 10 dimensions
            +: ad with broadcasting: adds the vector to each line of the matrix
            softmax(matrix) applies softmax on each line
            softmax (line) applies an exp to each value then divides by the
            Y: output matrix with 100 lines and 10 columns
'''


#x is data input  X: matrix for 100 grayscale images 28x28 pixels, flattened (there are 100 images in the mini batch)
X = tf.placeholder(tf.float32, [None, 28, 28, 1])

#Y_ is where the correct answers/labels will go
Y_ = tf.placeholder(tf.float32, [None, 10])

#weights W[784,10] 784=28*28  784x10 matrix with all zeroes that will be weights for each point in the 784,10 matrix
#784 is for each pixel and 10 is for each digit (0 - 9)

W = tf.Variable(tf.zeros([784,10]))

#biases b[10]
b = tf.Variable(tf.zeros([10])


#flatten the images into a single line of pixels
# -1 in the reshape definition means "the only possible dimension that will prese"
XX = tf.reshape(X, [-1,784])

#the model
Y = tf.nn.softmax(tf.matmul(XX, W) + b)

#loss function: cross-enropy = - sum( Y_i * log(Yi))
                                Y: the computed output vector
                                Y_: the desired output vector

#cross-entropy
#log takes the log of each element, * multiplies the tensors element by element
# reduce_mean will add all the components in the tensor
# so here we end up with the total cross entropy for all images in the batch

cross_entropy = -tf.reduce_mean(Y_ * tf.log(Y)) * 1000
#normalized for batch * 10 because



