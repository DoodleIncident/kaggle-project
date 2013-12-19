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

N = 70000
n = 7000

with open('train.csv', 'rb') as tf:
    r = csv.reader(tf, delimiter=',')
    all_fields = list(r)
    pre_input = map(lambda row: [row[1]], all_fields)
    pre_output = map(lambda row: row[13:28], all_fields)
    corpus = np.array([c[0] for c in pre_input[1:]])
    labels = np.array(pre_output[1:],float)

for idx in range(0,15):
	train_docs = corpus[0:N]
	train_labels = np.array(labels[0:N])
	test_docs = corpus[N:N+n]
	test_labels = np.array(labels[N:N+n])

	vect = CountVectorizer(min_df=100)
	tfidf = TfidfTransformer()
	# Construct a linear fit object
	lin_reg = LinearRegression()

	train_docs = vect.fit_transform(train_docs)
	train_docs = tfidf.fit_transform(train_docs)
	lin_reg.fit(train_docs, train_labels[:,idx])

	test_docs = vect.transform(test_docs)
	test_docs = tfidf.transform(test_docs)
	predicted = lin_reg.predict(test_docs)

	predicted = [fabs(x) for x in predicted]

	err = 0
	for i,j in zip(test_labels[:,idx],predicted):
		err += (j-i)**2
	err = (err / n) ** 0.5
	print err*100
