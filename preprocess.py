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
    pre_sentiment = map(lambda row: row[4:9], all_fields)
    pre_when = map(lambda row: row[9:13], all_fields)
    pre_weather = map(lambda row: row[13:], all_fields)

    corpus = [c[0] for c in pre_input]

    vectorizer = CountVectorizer(min_df=1600, max_df=64000)
    X = vectorizer.fit_transform(corpus)

    util.save_sparse_matrix('npy/input_tokens', X)
    numpy.save("npy/sentiment_layer.npy", numpy.array(pre_sentiment[1:],float))
    numpy.save("npy/when_layer.npy", numpy.array(pre_when[1:],float))
    numpy.save("npy/weather_layer.npy", numpy.array(pre_weather[1:],float))
