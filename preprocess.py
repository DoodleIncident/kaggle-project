import csv

from nltk import tokenize
from nltk.stem.lancaster import LancasterStemmer
st = LancasterStemmer()

import norvig

from collections import defaultdict

def escape_tokens(tweet):
    return tweet.replace("@mention", "AT_MENTION").replace("{link}", "BRACKET_LINK")

def tokenized(tweet):
    # Hah, these are just thrown away by the spellchecker anyway...
    #escaped = escape_tokens(tweet)

    lowered = tweet.lower()
    tokenized = tokenize.word_tokenize(lowered)
    spellchecked = [t for t in tokenized if norvig.known([t])]
    stemmed = [st.stem(s) for s in spellchecked]

    return stemmed

with open('train.csv', 'rb') as tf,\
        open('pre/input.csv', 'wb') as pi,\
        open('pre/output.csv', 'wb') as po:
        #open('pre/tweets.csv', 'wb') as pt: # NO. BAD. DO NOT DO THIS THING.
    r = csv.reader(tf, delimiter=',')
    wi = csv.writer(pi, delimiter=',')
    wo = csv.writer(po, delimiter=',')
    #wt = csv.writer(pt, delimiter=',')

    all_fields = list(r)

    # tweet, state, location
    pre_input = map(lambda row: [row[1]], all_fields)
    # s*, w*, k*
    pre_output = map(lambda row: row[4:], all_fields)

    token_set = set()
    tweet_tokens = []
    for s in pre_input:
        tweet_list = tokenized(s[0])
        tweet_tokens.append(tweet_list)
        token_set.update(set(tweet_list))

    all_tokens = list(token_set)

    print tweet_tokens[0:5]
    print "Because now we know there are", str(len(token_set)), "unique tokens in total"

    # row are tweets, cols are tokens
    pre_tokens = []
    pre_tokens.append(all_tokens) # Token list as a header

    for t_tweet in tweet_tokens:
        freqs = defaultdict(int, {t: t_tweet.count(t) for t in t_tweet})
        pre_tokens.append([freqs[t] for t in all_tokens])

    print len(pre_tokens), len(pre_tokens[0])

    for row_in, row_out in zip(pre_input, pre_output):
        wi.writerow(row_in)
        wo.writerow(row_out)

# we have a list of tokens as a set (no dups)

