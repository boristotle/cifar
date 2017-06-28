'''
inputs changing with time
activations of intermediate layers at each step will re-inject them as part of our input for the next step
internal state computed at each step then re-injected into the input
internal state also used to compute outputs via a softmax activation layer

RNN Training
same weights and biases are shared across iterations
we unroll the cell multiple times so we can change weights and biases
and therefore change the internal state at any time to match our expected output

we can also add layers

LSTM are the solutions to RNN
Long Short Term Memory

#input = real input + internal state
#inputs for one hot encoded alphabet is a vector looking like this [0,0,0,0,1] that represents each letter
concatenate: x = Xt | Ht-1   (vector size = p + n)

#internal vectors of values 0-1 all with different weights and biases
#all vectors inside of the cell are the same size n (forget gate, update gate, result gate, input, new C, new H)
forget gate: f = sigmoid(X.Wf + bf)
update gate: u = sigmoid(X.Wu + bu)
result gate: r = sigmoid(X.Wr + br)


input: X' = tanh(X.Wc + bc)  (note: the tanh could be replaced with a relu)

#maintain a second memory state called C
#new memory state equals forget gate * previous memory state + update gate * inputs
new C: Ct = f * Ct-1 + u * X'

#new internal state equals result gate * memory state
new H: Ht = r * tanh(Ct)

output:  (vector size m)
Yt = softmax(Ht.W + b)

'''

'''
GRU gated recurrent unit (2 gates instead of 3)

'''

import tensorflow as tf

ALPHASIZE = 98
CELLSIZE = 512
NLAYERS = 3
SEQLEN = 30

cell = tf.nn.rnn_cell.GRUCell(CELLSIZE) #defines weights and biases internally

mcell = tf.nn.rnn_cell.MultiRNNCell([cell]*NLAYERS, state_is_tuple=False) #creates a 3 layer cell


#unroll the cell
#X is a sequence of characters (if i feed it a sequence of 5 characters it will be unrolled 5 times)
Hr, H = tf.nn.dynamic_rnn(mcell, X, intial_state=Hin)


#softmax readout layer
#Hr [BATCHSIZE, SEQLEN, CELLSIZE]

Hf = tf.reshape(Hr, [-1, CELLSIZE])    #[BATCHSIZE * SEQLEN, CELLSIZE]
Ylogits = layers.linear(Hf, ALPHASIZE) #[BATCHSIZE * SEQLEN, ALPHASIZE]
Y = tf.nn.softmax(Ylogits)             #[BATCHSIZE * SEQLEN, ALPHASIZE]


loss = tf.nn.softmax_cross_entropy_with_logits(Ylogits, Y_)



#Placeholders and the rest

Xd = tf.placeholder(tf.uint8, [None, None])
X = tf.one_hot(X, ALPHASIZE, 1.0, 0.0)
Yd_ = tf.placeholder(tf.uint8, [None, None])
Y_ = tf.one_hot(Y_, ALPHASIZE, 1.0, 0.0)

Hin = tf.placeholder(tf.float32, [None, CELLSIZE*NLAYERS])

#Y, loss = my_model(Y, 1)

predictions = tf.argmax(Y, 1)
predictions = tf.reshape(predictions, [batchsize, -1])

train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)


#BATCHING must pass lines of characters


for x, y_ in tf.models.rnn.ptb.reader.ptb_iterator(codetext, BATCHSIZE, SEQLEN)