#How to use Tensorflow for Time Series:  https://www.youtube.com/watch?v=hhJIztWR_vo


#from Ipython.display import Image
#from Ipython.core.display import HTML
from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as py



#each epoch has a number of batches
num_epochs = 100
total_series_length = 50000

truncated_backprop_length = 15


#building a 3 layer recurrent network
#of neurons in hidden layer
state_size = 4

#data is binary (sequence to sequence mapping)
num_classes = 2

#
echo_step = 3

batch_size = 5
num_batches = total_series_length // batch_size // truncated_backprop_length

#STEP 1 (Collect Data)

def generateData():
    #0,1 50k samples
    x = np.array(np.random.choice(2, total_series_length, p=[0.5, 0.5]))
    #shift 3 step to the left
    y = np.roll(x, echo_step)
    y[0:echo_step] = 0
    x = x.reshape((batch_size, -1))
    y = y.reshape((batch_size, -1))

    return (x, y)

data = generateData()
print('data', data)


#STEP 2 (Build the model)

#this is what we feed our input data to
batchX_placeholder = tf.placeholder(dtype=tf.float32, shape=truncated_backprop_length, name='batchX_placeholder')
batchY_placeholder = tf.placeholder(dtype=tf.float32, shape=truncated_backprop_length, name='batchY_placeholder')

init_state = tf.placeholder(dtype=tf.float32, shape=[batch_size, state_size])

W = tf.Variable(initial_value=np.random.rand(state_size + 1, state_size), dtype=tf.float32)
b = tf.Variable(initial_value=np.zeros((1,state_size)), dtype=tf.float32)

W2 = tf.Variable(initial_value=np.random.rand(state_size + 1, num_classes), dtype=tf.float32)
b2 = tf.Variable(initial_value=np.zeros((1,num_classes)), dtype=tf.float32)


#unpack matrix into 1 dimensional array

inputs_series = tf.unstack(batchX_placeholder, axis=0)
labels_series = tf.unstack(batchY_placeholder, axis=0)


#logits is short for logistic transform, these are the output values that we are squashing (using softmax on) to give us our predictions
