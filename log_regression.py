import numpy
import scipy.sparse as sp
import theano
import theano.tensor as T

import util

rng = numpy.random

N = 35000

sparse_tokens = util.load_sparse_matrix("npy/input_tokens.npz").tolil()
input_tokens = sparse_tokens.todense()
tweet_tokens = input_tokens[:N,:]
test_tokens = input_tokens[N:2*N,:]
del input_tokens

feats = tweet_tokens.shape[1]

output_layer = numpy.load("npy/output_layer.npy")
is_cloudy = (output_layer[:N,9] > 0) + 0
test_is_cloudy = (output_layer[N:2*N,9] > 0) + 0
del output_layer

D = (tweet_tokens, is_cloudy)
C = (test_tokens, test_is_cloudy)

training_steps = 10000

x = T.matrix("x")
y = T.vector("y")
w = theano.shared(rng.randn(feats), name="w")
b = theano.shared(0., name="b")

#print "Initial model:"
#print w.get_value(), b.get_value()

p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))
prediction = p_1 > 0.5
xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1)
cost = xent.mean() + 0.01 * (w ** 2).sum()
gw,gb = T.grad(cost, [w, b])

train = theano.function(
            inputs=[x,y],
            outputs=[prediction, xent],
            updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))
predict = theano.function(inputs=[x], outputs=prediction)

for i in range(training_steps):
    pred, err = train(D[0], D[1])

#print "Final model:"
#print w.get_value(), b.get_value()
#print "target values for D:", D[1]
#print "prediction on D:", predict(D[0])

num_correct = len(["I" for actual, predicted in zip(predict(C[0]),C[1]) if actual == predicted])
print num_correct, '/', N
