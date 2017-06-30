#https://www.youtube.com/watch?v=Yl5lDaYvNqI&index=8&list=PLXO45tsB95cJHXaDKpbwr5fC_CCYylw1f
#placeholders in tensorflow

import tensorflow as tf

#args are type and shape (no shape is passed in this example)
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1, input2)


sess = tf.Session()
result = sess.run(output, feed_dict={input1: 2.0, input2: 4.0})
print('result', result)