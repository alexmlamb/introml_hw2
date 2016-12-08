
import theano
import theano.tensor as T
import numpy as np
import numpy.random as rng

def join2(a,b):
    return T.concatenate([a,b],axis=1)

def join3(a,b,c):
    return T.concatenate([a,b,c],axis=1)

W1 = theano.shared(0.01 * rng.normal(size = (512+10,512)).astype('float32'))
W2 = theano.shared(0.01 * rng.normal(size = (512,1)).astype('float32'))

b1 = theano.shared(np.zeros(shape = (512,)).astype('float32'))
b2 = theano.shared(np.zeros(shape = (1,)).astype('float32'))

o_s = T.vector()
h_s = T.matrix()
onehot = T.matrix()
#onehot = T.matrix()

h1 = T.dot(join2(h_s,onehot), W1) + b1
h1 = T.maximum(0.0, h1)
grad_oa_est = (T.nnet.softplus(T.dot(h1, W2) + b2)).flatten()

loss = T.mean(T.abs_(grad_oa_est - o_s))

updates = {}

import lasagne

updates = lasagne.updates.adam(loss, [W1, W2, b1, b2])

#for param in [W1,W2,b1,b2]:
#    updates[param] = param - 0.001 * T.grad(loss,param)

#grad_oa_est = T.clip(grad_oa_est,0.01,0.99)

#h_a, onehot -> grad_oa

#(128,512), (128,10) ==> (128,10)

predict_method = theano.function(inputs = [h_s,onehot], outputs = grad_oa_est)

train_method = theano.function(inputs = [o_s, h_s,onehot], outputs = loss, updates=updates)



