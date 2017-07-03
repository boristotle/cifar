#https://www.youtube.com/watch?v=e05zY-TJb5k&list=PLXO45tsB95cJHXaDKpbwr5fC_CCYylw1f&index=18
#save neural network for future use then use them using tf.train.Saver()

#SAVE TRAINED NETWORK
import tensorflow as tf

#save to file
#remember to define the same dtype and the same shape when restoring
W = tf.Variable([[1,2,3], [1,2,3]], dtype=tf.float32, name='weights')
b = tf.Variable([[1,2,3]], dtype=tf.float32, name='biases')

init = tf.global_variables_initializer()

#saver can only store the variables, can't store the shape so remember to define the same dtype and the same shape when restoring
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    save_path = saver.save(sess, "./save_net.ckpt")
    print("Save to path", save_path)








#RESTORE SAVED NETWORK
import tensorflow as tf
###restore variables for use again with more test data########
#redefine the same shape and dtype for your variables

#shape is two rows, three columns
W = tf.Variable(tf.zeros([2,3]), dtype=tf.float32, name='weights')

#shape is one row, three columns
b = tf.Variable(tf.zeros([1,3]), dtype=tf.float32, name='biases')

#don't need init step
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, "./save_net.ckpt")
    print('weights:', sess.run(W))
    print('biases:', sess.run(b))