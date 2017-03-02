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

m = 784

def init_params():

    params = [0]*4


    params[0] = 0.05 * rng.normal(size = (784,m)).astype('float32')
    params[1] = 0.05 * rng.normal(size = (m, m)).astype('float32')
    params[2] = 0.05 * rng.normal(size = (m, m)).astype('float32')
    params[3] = 0.05 * rng.normal(size = (m, 10)).astype('float32')

    return params

#This is for feedback alignment method.  
def init_params_fa():

    params = [0]*4

    params[0] = 0.01 * rng.normal(size = (10,m)).astype('float32')
    #params[1] = 0.01 * rng.normal(size = (10,m)).astype('float32')
    params[1] = params[0]
    params[2] = params[0]
    #params[2] = 0.01 * rng.normal(size = (10,m)).astype('float32')
    params[3] = 0.01 * rng.normal(size = (10,10)).astype('float32')

    #params['W2'] = params['W1'].T
    #params['W2'] =np.maximum(-2.0,rng.normal(size = (m, 10))).astype('float32')

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

    inp = x
    h_a_lst = []
    h_s_lst = [x]

    for j in range(0,3):
        h_a_lst.append(np.dot(h_s_lst[-1], p[j]))
        h_s_lst.append(np.maximum(0.0, h_a_lst[-1]))

    h_s = h_s_lst[-1]

    o_a = np.dot(h_s, p[-1])
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

    grad_W2 = np.dot(h_s.T, grad_oa)

    p[0] -= 0.001 * np.dot(h_s_lst[0].T, np.dot(grad_oa, bfa[0]) * (h_a_lst[0] >= 0))
    p[1] -= 0.001 * np.dot(h_s_lst[1].T, np.dot(grad_oa, bfa[1]) * (h_a_lst[1] >= 0))
    p[2] -= 0.001 * np.dot(h_s_lst[2].T, np.dot(grad_oa, bfa[2]) * (h_a_lst[2] >= 0))

    p[3] -= 0.001 * grad_W2





