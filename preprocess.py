import csv
from sklearn.feature_extraction.text import CountVectorizer

import numpy
import util

with open('train.csv', 'rb') as tf:
    r = csv.reader(tf, delimiter=',')
    # wi = csv.writer(pi, delimiter=',')
    # wo = csv.writer(po, delimiter=',')

    all_fields = list(r)

    pre_input = map(lambda row: [row[1]], all_fields)
    pre_output = map(lambda row: row[4:], all_fields)

    corpus = [c[0] for c in pre_input]

    vectorizer = CountVectorizer(min_df=1600, max_df=64000)
    X = vectorizer.fit_transform(corpus)

    util.save_sparse_matrix('npy/input_tokens', X)
    numpy.save("npy/output_layer.npy", numpy.array(pre_output[1:],float))

    # for row_in, row_out in zip(pre_input, pre_output):
    #     wi.writerow(row_in)
    #     wo.writerow(row_out)
