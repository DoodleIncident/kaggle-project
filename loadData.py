import csv
import numpy
import theano
import theano.tensor as T
rng = numpy.random

def sentiment(row):
    return tuple([float(row[sk]) for sk in s_keys])

tweets = []
outputs = []

with open('train.csv', 'rb') as tf:
    r = csv.reader(tf, delimiter=',')
    r.next() # Skip header
    l = list(r)

