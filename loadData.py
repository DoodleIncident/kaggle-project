import csv
import numpy

with open('train.csv', 'rb') as csvfile:
	reader = csv.reader(csvfile, delimiter=',')
	reader.next()
	tweets = list(reader)
	for tweet in tweets:
		a = map(float, tweet[4:9])
		if sum(a) != 1: print tweet[0]
	# print len(list(reader))
