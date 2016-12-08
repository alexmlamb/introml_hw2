import theano
import theano.tensor as T

def crossent(p,y): 
    return -T.mean(T.log(p)[T.arange(y.shape[0]), y])

def accuracy(p,y):
    return T.mean(T.eq(T.argmax(p, axis = 1),y))

if __name__ == "__main__":

    p = T.matrix()
    y = T.ivector()

    f = theano.function([p,y], [crossent(p,y), accuracy(p,y)], allow_input_downcast=True)

    print f([[0.9,0.1]], [0])
    print f([[0.7,0.3]], [0])
    print f([[0.5,0.5]], [0])
    print f([[0.1,0.9]], [0])


