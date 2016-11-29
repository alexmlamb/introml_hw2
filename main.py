'''
Basically implement a numpy NN for mnist.  

Fun twist - also try feedback alignment.  

Even funner twist - try "online direct feedback alignment".  
'''

import numpy as np
import numpy.random as rng
import gzip
import cPickle as pickle
import random

mn = gzip.open("/u/lambalex/data/mnist/mnist.pkl.gz")

train, valid, test = pickle.load(mn)

trainx,trainy = train
validx,validy = valid

trainy = trainy.astype('int32')
validy = validy.astype('int32')


def init_params():

    params = {}

    params['W1'] = 0.05 * rng.normal(size = (784,512)).astype('float32')
    params['W2'] = 0.05 * rng.normal(size = (512, 10)).astype('float32')

    #params['b1'] = np.zeros(shape = (512,)).astype('float32')
    #params['b2'] = np.zeros(shape = (10,)).astype('float32')

    return params

#This is for feedback alignment method.  
def init_params_fa():
    pass

p = init_params()
p_fa = init_params_fa()

for iteration in range(0,10000):

    ###
    #Forward Prop
    #given x,y
    ###

    r = random.randint(0,40000)

    x = trainx[r:r+128]
    y = trainy[r:r+128].flatten()


    h_a = np.dot(x, p['W1'])
    h_s = np.maximum(0.0, h_a)

    o_a = np.dot(h_s, p['W2'])
    o_a_exp = np.exp(o_a - o_a.max(axis=1, keepdims=True))
    o_s = o_a_exp / np.sum(o_a_exp, axis = 1, keepdims=True)
    

    accuracy = (o_s.argmax(axis=1) == y).mean()
    
    if iteration % 100 == 0:
        print accuracy


    ###
    #Backward Prop
    ###

    onehot = np.zeros(shape=(128,10)).astype('float32')
    onehot[np.arange(y.shape[0]),y] = 1.0

    grad_oa = o_s - onehot

    grad_hs = np.dot(grad_oa, p['W2'].T)

    grad_ha = grad_hs

    grad_ha[h_a<=0] = 0.0

    grad_x = np.dot(grad_ha, p['W1'].T)

    #print grad_x[0]

    ###
    #Parameter gradients and update parameters
    ###

    grad_W1 = np.dot(x.T, grad_ha)

    grad_W2 = np.dot(h_s.T, grad_oa)

    p['W1'] -= grad_W1 * 0.001
    p['W2'] -= grad_W2 * 0.001



