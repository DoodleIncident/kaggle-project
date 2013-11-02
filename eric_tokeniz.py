from nltk.tokenize import RegexpTokenizer
from collections import Counter
import csv
import norvig


spellcheck = norvig.correct

# change sampleSize as desired
sampleSize = 90000
count = 1

inFile = open('train.csv', 'rb')
reader = csv.reader(inFile)

uniqueTokens = dict()
numTokens = 0
tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
rowNum = 0
tweet_tokens = []
for row in reader:
	if (rowNum == 0):
		rowNum = rowNum + 1
		continue
	print 'tweet#', rowNum
	# tokenize each tweet
	# tweet_tokens.append( tokenizer.tokenize(row[1].lower()))
	
	corrected = list()
	for x in tokenizer.tokenize(row[1].lower()):
		corrected.append(spellcheck(x))

	tweet_tokens.append( corrected)



	# add each token to the unique collection
	for t in tweet_tokens[ - 1]:
		uniqueTokens[t] = uniqueTokens.get(t, 0)
		numTokens = numTokens + 1
	rowNum = rowNum + 1
	if(rowNum > sampleSize):
		break

print'Number of tweets read:', len(tweet_tokens)
# for tweet in tweet_tokens:
# 	print tweet,'\n'

print 'Number of uniqueTokens:', len(uniqueTokens), '\n'

avgTokens = numTokens / len(tweet_tokens)


print 'Total number of tokens:', numTokens
print 'Average tokens/tweet:', avgTokens
# print 'Times "rain" has been said:', uniqueTokens["rain"]