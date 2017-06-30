#https://www.youtube.com/watch?v=UYyqNH3r4lk&list=PLXO45tsB95cJHXaDKpbwr5fC_CCYylw1f&index=7
#variables in tensorflow


import tensorflow as tf

state = tf.Variable(0, name='counter')
#print(state.name)
one = tf.constant(1)
new_value = tf.add(state, one)
updated_state = tf.assign(state, new_value)

print('updated_state', updated_state)


init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
for i in range(3):
    sess.run(updated_state)
    print(sess.run(state))
