import csv
from nltk import tokenize
from nltk.stem.lancaster import LancasterStemmer
st = LancasterStemmer()

def escape_tokens(tweet):
    return tweet.replace("@mention", "AT_MENTION").replace("{link}", "BRACKET_LINK")

def tokenized(tweet):
    lowered = tweet.lower()
    escaped = escape_tokens(lowered)
    stripped = escaped.strip(".,")
    stemmed = st.stem(stripped)
    return tokenize.word_tokenize(stemmed)


with open('train.csv', 'rb') as tf,\
        open('pre/input.csv', 'wb') as pi,\
        open('pre/output.csv', 'wb') as po,\
        open('pre/tweets.csv', 'wb') as pt:
    r = csv.reader(tf, delimiter=',')
    wi = csv.writer(pi, delimiter=',')
    wo = csv.writer(po, delimiter=',')
    wt = csv.writer(pt, delimiter=',')

    all_fields = list(r)

    # tweet, state, location
    pre_input = map(lambda row: [row[1]], all_fields)
    # s*, w*, k*
    pre_output = map(lambda row: row[4:], all_fields)

    all_tokens = set()
    tweet_tokens = []
    for s in pre_input:
        tweet_set = set(tokenized(escape_tokens(s[0])))
        tweet_tokens.append(tweet_set)
        all_tokens.update(tweet_set)

    print tweet_tokens[0:5]
    print "Because now we know there are", str(len(all_tokens)), "unique tokens in total"
    
    #for row_in, row_out in zip(pre_input, pre_output):
        #wi.writerow(row_in)
        #wo.writerow(row_out)

