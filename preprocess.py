import csv
from sklearn.feature_extraction.text import CountVectorizer

import numpy
import util

with open('train.csv', 'rb') as tr, open('test.csv', 'rb') as te:
    r1 = csv.reader(tr, delimiter=',')
    r2 = csv.reader(te, delimiter=',')
    # wi = csv.writer(pi, delimiter=',')
    # wo = csv.writer(po, delimiter=',')

    train_fields = list(r1)[1:]
    test_fields = list(r2)[1:]

    pre_input = map(lambda row: [row[1]], train_fields)
    pre_sentiment = map(lambda row: row[4:9], train_fields)
    pre_when = map(lambda row: row[9:13], train_fields)
    pre_weather = map(lambda row: row[13:], train_fields)

    train_corpus = [c[0] for c in pre_input]
    test_corpus = [c[1] for c in test_fields]

    vectorizer = CountVectorizer(min_df=100)
    X = vectorizer.fit_transform(train_corpus)
    Y = vectorizer.transform(test_corpus)

    print "First everything:", train_fields[0]
    print "First tweet:", pre_input[0]
    print "First sentiment:", pre_sentiment[0]
    print "First when:", pre_when[0]
    print "First weather:", pre_weather[0]
    print "First vectorized:", X[0]
    print X.shape
    #print vectorizer.get_feature_names()

    util.save_sparse_matrix('npy/input_tokens', X)
    util.save_sparse_matrix('npy/test_tokens', Y)

    numpy.save("npy/sentiments_layer.npy", numpy.array(pre_sentiment,float))
    numpy.save("npy/whens_layer.npy", numpy.array(pre_when,float))
    # I guess I'm not doing anything really with this
    #numpy.save("npy/weather_layer.npy", numpy.array(pre_weather,float))
