import csv
import util
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression
from math import fabs

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

A = np.ones(len(test_input))
offset = 10000
for idx in range(0,15):
    B = np.zeros(len(test_input))
    for jdx in range(0,len(corpus)/offset):
        train_docs = corpus
        train_labels = np.array(labels)
        test_docs = test_input

        vect = CountVectorizer(min_df=100)
        tfidf = TfidfTransformer()
        lin_reg = LinearRegression()

        train_docs = vect.fit_transform(train_docs[jdx*offset:jdx*offset+offset])
        train_docs = tfidf.fit_transform(train_docs)
        tl = train_labels[:,idx]
        lin_reg.fit(train_docs, tl[jdx*offset:jdx*offset+offset])

        test_docs = vect.transform(test_docs)
        test_docs = tfidf.transform(test_docs)
    	predicted = lin_reg.predict(test_docs)

    	predicted = np.array([(x+fabs(x))/2 for x in predicted])
        B += predicted

    B = B/(len(corpus)/offset)
    A = np.c_[ A, B ]

A = np.array(A[:,1:])
np.savetxt('out/kinds.csv',A,delimiter=',',fmt='%.3f')
