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



channels_layer1 = 200
channels_layer2 = 100
channels_layer3 = 60
output_layer4 = 30
output_neurons = 10

#3 is for color image with 3 channels (R,G,B)
X = tf.placeholder(tf.float32, [None, 32, 32, 3])

W1 = tf.Variable(tf.truncated_normal([32*32,channels_layer1], stddev=0.1))
B1 = tf.Variable(tf.zeros([channels_layer1]))

W2 = tf.Variable(tf.truncated_normal([channels_layer1,channels_layer2], stddev=0.1))
B2 = tf.Variable(tf.zeros([channels_layer2]))

W3 = tf.Variable(tf.truncated_normal([channels_layer2,channels_layer3], stddev=0.1))
B3 = tf.Variable(tf.zeros([channels_layer3]))

W4 = tf.Variable(tf.truncated_normal([channels_layer3,output_layer4], stddev=0.1))
B4 = tf.Variable(tf.zeros([output_layer4]))

W5 = tf.Variable(tf.truncated_normal([output_layer4,output_neurons]))
B5 = tf.Variable(tf.zeros([output_neurons]))


#-1 means only one correct answer, 32x32 is image dimensions
X = tf.reshape(X, [-1, 32*32])
print(tf.shape(X))

Y1 = tf.nn.relu(tf.matmul(X, W1) + B1)
Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2)
Y3 = tf.nn.relu(tf.matmul(Y2, W3) + B3)
Y4 = tf.nn.relu(tf.matmul(Y3, W4) + B4)
Y = tf.nn.softmax(tf.matmul(Y4, W5) + B5)


#placeholder for correct answers
Y_ = tf.placeholder(tf.float32, [None, 10])

#loss function (comparing our predictions to known labels)
cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y))


'''
is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = -tf.reduce_mean(tf.cast(is_correct, tf.float32))
'''

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
