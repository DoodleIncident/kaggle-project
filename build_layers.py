import csv

from nltk import tokenize
from nltk.stem.lancaster import LancasterStemmer
st = LancasterStemmer()
import norvig

from collections import defaultdict

import numpy
import scipy.sparse as sp

def save_sparse_matrix(filename,x):
    x_coo=x.tocoo()
    row=x_coo.row
    col=x_coo.col
    data=x_coo.data
    shape=x_coo.shape
    numpy.savez(filename,row=row,col=col,data=data,shape=shape)

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

output_layer = numpy.loadtxt(open('pre/output.csv','rb'), delimiter=',', skiprows=1)

print output_layer.shape
print output_layer[0]

numpy.save("npy/output_layer.npy", output_layer)

with open('pre/input.csv', 'rb') as pi:
    r = csv.reader(pi, delimiter=',')
    pre_input = list(r)

    token_set = set()
    tweet_tokens = []
    for s in pre_input:
        tweet_list = tokenized(s[0])
        tweet_tokens.append(tweet_list)
        token_set.update(set(tweet_list))

    all_tokens = numpy.array(list(token_set))

    numpy.save("npy/all_tokens.npy", all_tokens)

    print tweet_tokens[0:5]
    print "Number of tokens:", str(len(token_set))

    # row are tweets, cols are tokens
    input_tokens = numpy.empty((len(pre_input),len(all_tokens)))

    for i, t_tweet in enumerate(tweet_tokens):
        freqs = defaultdict(int, {t: t_tweet.count(t) for t in t_tweet})
        input_tokens[i] = [freqs[t] for t in all_tokens]

    print len(input_tokens), len(input_tokens[0])

    print input_tokens.shape
    print input_tokens[0]

    sparse_tokens = sp.lil_matrix(input_tokens)

    #numpy.save("npy/input_tokens.npy", input_tokens)
    save_sparse_matrix("npy/input_tokens", sparse_tokens)
