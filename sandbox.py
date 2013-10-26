import csv
import numpy
import theano
import theano.tensor as T
rng = numpy.random

tweets = []
outputs = []

def preprocess():
    with open('train.csv', 'rb') as tf, open('pre/input.csv', 'wb') as pi, open('pre/output.csv', 'wb') as po:
        r = csv.reader(tf, delimiter=',')
        wi = csv.writer(pi, delimiter=',')
        wo = csv.writer(po, delimiter=',')

        all_fields = list(r)

        # tweet, state, location
        pre_input = map(lambda row: row[:1] + row[1:4], all_fields)
        # s*, w*, k*
        pre_output = map(lambda row: row[:1] + row[4:], all_fields)
        
        for row_in, row_out in zip(pre_input, pre_output):
            wi.writerow(row_in)
            wo.writerow(row_out)

def load_output():
    sentiments = numpy.loadtxt(open('pre/output.csv', 'rb'),delimiter=',',skiprows=1)

load_output()
