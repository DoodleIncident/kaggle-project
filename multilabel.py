import csv
import util
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVR

N = 1000

with open('train.csv', 'rb') as tf:
    r = csv.reader(tf, delimiter=',')
    all_fields = list(r)
    pre_input = map(lambda row: [row[1]], all_fields)
    pre_output = map(lambda row: row[4:], all_fields)

    corpus = np.array([c[0] for c in pre_input])

train_docs = corpus[0:N]
pre_output = np.array(pre_output[1:],float)
pre_output = np.array([o[5:9] for o in pre_output.tolist()]).tolist()
train_labels = util.sparser(pre_output[0:N])
test_docs = corpus[N:N+100]
# target_names = ['clouds', 'cold', 'dry', 'hot', 'humid', 'hurricane', 'I can\'t tell', 'ice', 'other', 'rain', 'snow', 'storms', 'sun', 'tornado', 'wind']
target_names = ['current (same day) weather', 'future (forecast)', 'I can\'t tell', 'past weather']
# target_names = ['I can\'t tell', 'Negative', 'Neutral / author is just sharing information', 'Positive', 'Tweet not related to weather condition']

# min_df=800,max_df=72000
classifier = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(LinearSVC()))])
classifier.fit(train_docs, train_labels)
predicted = classifier.predict(test_docs)
for item, labels in zip(test_docs, predicted):
    print '%s => %s' % (item, ', '.join(target_names[x] for x in labels))
