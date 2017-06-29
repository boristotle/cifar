import tensorflow as tf
import pickle


meta_file = open('cifar-10-batches-py/batches.meta', "rb")
dict = pickle.load(meta_file)

label_names = dict["label_names"]
#print('dict', dict["label_names"])



img_data_file = open('cifar-10-batches-py/data_batch_1', "rb")
img_dict = pickle.load(img_data_file, encoding="latin1")
#print('img_data', img_dict['data'])
#print('img_labels', img_dict['labels'])

train_features = img_dict['data'].reshape((len(img_dict['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
train_labels = img_dict['labels']

print('train_features', len(train_features))



test_data_file = open('cifar-10-batches-py/test_batch', "rb")
test_dict = pickle.load(test_data_file, encoding="latin1")

test_features = test_dict['data']
test_labels = test_dict['labels']



channels_layer1 = 4
channels_layer2 = 8
channels_layer3 = 12
output_neurons = 10

#3 is for color image with 3 channels (R,G,B)
X = tf.placeholder(tf.float32, [None, 32, 32, 3])

#STRIDE = 1 ([32x32, 4])
W1 = tf.Variable(tf.truncated_normal([5,5,1,channels_layer1], stddev=0.1))
B1 = tf.Variable(tf.ones([channels_layer1]))

#STRIDE = 2 ([16x16, 8])
W2 = tf.Variable(tf.truncated_normal([4,4,channels_layer1,channels_layer2], stddev=0.1))
B2 = tf.Variable(tf.ones([channels_layer2]))

#STRIDE = 2 ([8x8, 12])
W3 = tf.Variable(tf.truncated_normal([4,4,channels_layer2,channels_layer3], stddev=0.1))
B3 = tf.Variable(tf.ones([channels_layer3]))

output_layer4 = 200

#([8x8x12, 200])
W4 = tf.Variable(tf.truncated_normal([8*8*channels_layer3,output_layer4], stddev=0.1))
B4 = tf.Variable(tf.ones([output_layer4]))


#([200, 10])
W5 = tf.Variable(tf.truncated_normal([output_layer4,output_neurons]))
B5 = tf.Variable(tf.zeros([output_neurons]))


#-1 means only one correct answer, 32x32 is image dimensions
X = tf.reshape(X, [-1, 32*32])


Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding='SAME') + B1)
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1,2,2,1], padding='SAME') + B2)
Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1,2,2,1], padding='SAME') + B3)

#flatten all values of the Y3 layer to fully connected layer YY
YY = tf.reshape(Y3, shape=[-1, 8*8*12])


Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)
Y = tf.nn.softmax(tf.matmul(Y4, W5) + B5)


#placeholder for correct answers
Y_ = tf.placeholder(tf.float32, [None, 10])

#loss function (comparing our predictions to known labels)
cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y))


#TRAIN
optimizer = tf.train.GradientDescentOptimizer(0.003)
train_step = optimizer.minimize(cross_entropy)

#Run Training Loop

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

train_data = {X: train_features, Y_: train_labels}

sess.run(train_step, feed_dict=train_data)

a,c = sess.run([accuracy, cross_entropy], feed=train_data)

#test_data = {X: test_features, Y_: test_labels}
#a,c = sess.run([accuracy, cross_entropy], feed=test_data)
