import csv
import util
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

with open('train.csv', 'rb') as train,\
open('test.csv', 'rb') as test:
    r1 = csv.reader(train, delimiter=',')
    all_fields = list(r1)
    train_input = map(lambda row: [row[1]], all_fields)
    train_output = map(lambda row: row[13:28], all_fields)
    corpus = np.array([c[0] for c in train_input[1:]])
    labels = np.array(train_output[1:],float)

    r2 = csv.reader(test, delimiter=',')
    all_fields = list(r2)
    test_input = map(lambda row: [row[1]], all_fields)
    test_input = np.array([c[0] for c in test_input[1:]])

train_docs = corpus
train_labels = util.sparser(labels)
test_docs = test_input

vect = CountVectorizer(min_df=100)
tfidf = TfidfTransformer()
clf = OneVsRestClassifier(LinearSVC())

train_docs = vect.fit_transform(train_docs)
train_docs = tfidf.fit_transform(train_docs)
clf.fit(train_docs, train_labels)

test_docs = vect.transform(test_docs)
test_docs = tfidf.transform(test_docs)
predicted = clf.predict(test_docs)

predicted = util.list_of_tuples_to_2d_list(predicted)
predicted = np.array(util.denser(predicted, 15))
np.savetxt('out/kinds.csv',predicted,delimiter=',',fmt='%.3f')
