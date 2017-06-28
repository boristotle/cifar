#convolutional network uses the same weights for each neuron
#just need to create multiple output layers that use different weights


#filter size is 4x4, input channels is 3, 2 output channels (passes on the image)
#W[4,4,3,2]

'''
28x28x1  (original image)

convolutional_layer 1, 4 channels
W1[5,5,1,4]   patches of 5x5 pixels, 1 input channel/layer bc it is a greyscale image, 4 output channels/layers
[stride is 1 so we are still 28x28 pixels]
so the result is a 28x28x4 matrix (convolutional layer)

convolutional_layer 2, 8 channels
W2[4,4,4,8]  patches of 4x4 pixels, 4 input channels/layers (bc previous output was 4 channels/layers), 8 output channels
[stride is 2 so we are 14x14 pixels now]
so the result is a 14x14x8 matrix (convolutional layer)


convolutional_layer 3, 12 channels
W3[4,4,8,12] patches of 4x4 pixels, 8 input channels/layers (bc previous output was 8 channels/layers, 12 output channels
[stride on this matrix is 2 so we have 7x7 pixels now
so the result is a 7x7x12 matrix (convolutional layer)


convolutional_layer 4
fully connected layer W4[7x7x12, 200]
each neuron does a weighted sum of all the values of the values in the previous layer

convolutional_layer 5, output layer
softmax readout layer W5[200,10]

'''

#need a weights matrix and bias vector per layer

K = 4
L = 8
M = 12

W1 = tf.Variable(tf.truncated_normal([5,5,1,K], stddev=0.1))
B1 = tf.Variable(tf.ones([K]/10)
W2 = tf.Variable(tf.truncated_normal([4,4,K,L], stddev=0.1))
B2 = tf.Variable(tf.ones([L]/10)
W3 = tf.Variable(tf.truncated_normal([4,4,L,M], stddev=0.1))
B3 = tf.Variable(tf.ones([M]/10)

N = 200

W4 = tf.Variable(tf.truncated_normal([7*7*M, N], stddev=0.1))
B4 = tf.Variable(tf.ones([N]/10)
W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
B5 = tf.Variable(tf.zeros([10]))


#the model

#x[100,28,28,1]

Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1,1,1,1] padding='SAME') + B1)
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1,2,2,1] padding='SAME') + B2)
Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1,2,2,1] padding='SAME') + B3) #y3[100,7,7,12]

#flatten all values for fully connected layer
YY = tf.reshape(Y3, shape=[-1,7*7*M]) #yy[100,7*7*12]

Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)
Y5 = tf.nn.softmax(tf.matmul(Y4, W5) + B5)



'''the above causes overfitting, to fix we allow more degrees of freedom by using bigger pixel patches and more channels
so we have
layer 1 W1[6,6,1,6] stride 1 (6 channels)
layer 2 W2[5,5,6,12] stride 2 (12 channels)
layer 3 W3[4,4,12,24] stride 2 (24 channels)
fully connected layer W4[7x7x24, 200]
softmax readout layer W5[200,10]

we can also add dropout to the fully connected layer
''''

