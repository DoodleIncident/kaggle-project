import csv
import util
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

N = 70000
n = 7000

with open('train.csv', 'rb') as tf:
    r = csv.reader(tf, delimiter=',')
    all_fields = list(r)
    pre_input = map(lambda row: [row[1]], all_fields)
    pre_output = map(lambda row: row[13:], all_fields)
    corpus = np.array([c[0] for c in pre_input[1:]])
    labels = np.array(pre_output[1:],float)

train_docs = corpus[0:N]
train_labels = util.sparser(labels[0:N])
test_docs = corpus[N:N+n]
test_labels = util.sparser(labels[N:N+n])
target_names = ['clouds', 'cold', 'dry', 'hot', 'humid', 'hurricane', 'I can\'t tell', 'ice', 'other', 'rain', 'snow', 'storms', 'sun', 'tornado', 'wind']

# min_df=1600,max_df=64000
classifier = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(LinearSVC()))])
classifier.fit(train_docs, train_labels)
predicted = classifier.predict(test_docs)
#for item, labels in zip(test_docs, predicted):
    #print '%s => %s' % (item, ', '.join(target_names[x] for x in labels))
predicted = util.list_of_tuples_to_2d_list(predicted)
accuracy = util.compute_kind_error(test_labels, predicted)
accuracy = (100*accuracy) / (15*n)
print accuracy
