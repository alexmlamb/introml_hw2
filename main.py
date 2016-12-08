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

    m=512

    params['W1'] = 0.05 * rng.normal(size = (784,m)).astype('float32')
    params['W2'] = 0.05 * rng.normal(size = (m, 10)).astype('float32')

    #params['b1'] = np.zeros(shape = (512,)).astype('float32')
    #params['b2'] = np.zeros(shape = (10,)).astype('float32')

    return params

#This is for feedback alignment method.  
def init_params_fa():

    params = {}

    m=512

    params['W1'] = 0.0 * np.maximum(-2.0,rng.normal(size = (784,m))).astype('float32')
    params['W2'] = 0.0 * np.maximum(-2.0,rng.normal(size = (m, 10))).astype('float32')

    return params

p = init_params()
pfa = init_params_fa()

for iteration in xrange(0,1000000):

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
    
    if iteration % 1000 == 0:
        print accuracy


    ###
    #Backward Prop
    ###

    bfa = pfa
    #bfa = p

    onehot = np.zeros(shape=(x.shape[0],10)).astype('float32')
    onehot[np.arange(y.shape[0]),y] = 1.0

    grad_oa = o_s - onehot

    grad_hs = np.dot(grad_oa, bfa['W2'].T)

    grad_ha = grad_hs

    grad_ha[h_a<=0] = 0.0

    grad_x = np.dot(grad_ha, bfa['W1'].T)

    #print grad_x[0]

    ###
    #Parameter gradients and update parameters
    ###

    #ha
    grad_W1 = np.dot(x.T, grad_ha)

    #oa
    grad_W2 = np.dot(h_s.T, grad_oa)

    p['W1'] -= grad_W1 * 0.001
    p['W2'] -= grad_W2 * 0.001



