import csv
import util
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier

N = 70000
n = 7000

with open('train.csv', 'rb') as tf:
    r = csv.reader(tf, delimiter=',')
    all_fields = list(r)
    pre_input = map(lambda row: [row[1]], all_fields)
    pre_output = map(lambda row: row[13:28], all_fields)
    corpus = np.array([c[0] for c in pre_input[1:]])
    labels = np.array(pre_output[1:],float)

#target_names = ['clouds', 'cold', 'dry', 'hot', 'humid', 'hurricane', 'I can\'t tell', 'ice', 'other', 'rain', 'snow', 'storms', 'sun', 'tornado', 'wind']
train_docs = corpus[0:N]
train_labels = util.sparser(labels[0:N])
test_docs = corpus[N:N+n]
test_labels_bin = util.sparser(labels[N:N+n])
test_labels = labels[N:N+n]

vect = CountVectorizer(min_df=100)
tfidf = TfidfTransformer()
clf = OneVsRestClassifier(SGDClassifier(loss='perceptron'))

train_docs = vect.fit_transform(train_docs)
train_docs = tfidf.fit_transform(train_docs)
clf.fit(train_docs, train_labels)

test_docs = vect.transform(test_docs)
test_docs = tfidf.transform(test_docs)
predicted = clf.predict(test_docs)

#ngram_range=(2,2),min_df=1600,max_df=64000
# classifier = Pipeline([
#     ('vect', CountVectorizer(ngram_range=(1, 2))),
#     ('tfidf', TfidfTransformer()),
#     ('clf', OneVsRestClassifier(LinearSVC()))])
# classifier.fit(train_docs, train_labels)
# predicted = classifier.predict(test_docs)

#for item, labels in zip(test_docs, predicted):
    #print '%s => %s' % (item, ', '.join(target_names[x] for x in labels))

predicted = util.list_of_tuples_to_2d_list(predicted)
accuracy = util.compute_bin_error(test_labels_bin, predicted, 15)
accuracy = 100 - ((100*accuracy) / (15*n))
print accuracy

predicted = np.array(util.denser(predicted, 15))
error = util.compute_error(test_labels, predicted)
error = 100 * ((error / (15*n)) ** 0.5)
print error

print predicted