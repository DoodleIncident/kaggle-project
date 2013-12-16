import cPickle
import csv
import theano
import theano.tensor as T
import numpy

import util
from tutorial.code import mlp

def predict(filename):
    f = file('mdl/'+filename+'.save', 'rb')
    params = cPickle.load(f)
    f.close
    W_1, b_1, W_2, b_2 = params

    n_x = W_1.shape.eval()[0]
    n_h = b_1.shape.eval()[0]
    n_y = b_2.shape.eval()[0]

    test_input = util.load_sparse_matrix('npy/test_tokens.npz').todense()
    test_x = theano.shared(numpy.asarray(test_input,
            dtype=theano.config.floatX),
        borrow=True)

    x = T.matrix('x')

    rng = numpy.random.RandomState(1234)

    classifier = mlp.MLP(rng=rng, input=test_x,
            n_in=n_x, n_hidden=n_h, n_out=n_y, params=params)

    #classifier.hiddenLayer.W = W_1
    #classifier.hiddenLayer.b = b_1

    #classifier.logRegressionLayer.W = W_2
    #classifier.logRegressionLayer.b = b_2

    t_predict = theano.function(inputs=[],
                                outputs=classifier.predict(),
                                givens={ x: test_x })

    predictions = t_predict()
    numpy.savetxt('out/'+filename+'.csv', predictions,
            delimiter=',', fmt='%.3f')

predict("sentiments")
predict("whens")
