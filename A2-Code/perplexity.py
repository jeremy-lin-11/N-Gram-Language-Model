# tokenization, compute sentence probabilities
#   unkify_OOV_words: the purpose of <UNK> is to generalize 
#   to infrequent words in your dev/test set: if not in 
#   the 26602 vocab words, replace w/ UNK before computing probability

# this thing should take the ngram model we made from train and use those counts in calculating the MLE for test/dev sets
# tokenize test set, counts and all that of ngrams, use perplexity
# but if we're comparing test p(xi) in perplexity, 
# and word shows up we havent seen before? it would *0 or log0 and fuck up the perplexity formula?

class perplexity(object):
    def __init__(self):
        pass

    def perplexity(self, file_path, unkify_OOV_words=True):
        return