from tutorial.code import mlp, logistic_sgd

import util
import numpy
import theano
import theano.tensor as T

def shared_dataset(data_xy, borrow=True):
    data_x, data_y = data_xy

    #data_y = data_y.argmax(axis=1)
    #print data_y

    shared_x = theano.shared(numpy.asarray(data_x,
                                            dtype=theano.config.floatX),
                                borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                            dtype=theano.config.floatX),
                                borrow=borrow)
    #return shared_x, T.cast(shared_y, 'int32')
    return shared_x, shared_y

def load_tweets(dataset):
    TRAIN_N = 50000
    VALID_N = 10000 + TRAIN_N
    # test on the rest
    
    print "Loading", dataset

    input_layer = util.load_sparse_matrix("npy/input_tokens.npz").todense()
    output_layer = numpy.load("npy/" + dataset + '_layer.npy')

    train_set = input_layer[:TRAIN_N], output_layer[:TRAIN_N]
    valid_set = input_layer[TRAIN_N:VALID_N], output_layer[TRAIN_N:VALID_N]
    test_set = input_layer[VALID_N:], output_layer[VALID_N:]

    return [shared_dataset(train_set)
           ,shared_dataset(valid_set)
           ,shared_dataset(test_set)]

if __name__ == '__main__':
    mlp.test_mlp(load_data=load_tweets, dataset='sentiments')
    mlp.test_mlp(load_data=load_tweets, dataset='whens',use_old=True)
