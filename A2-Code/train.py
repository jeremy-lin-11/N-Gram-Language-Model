# TO DO:
# tokenization
#   already ‘done’, you just need to reverse the operation
#   Can be done for an individual line with a combination of .split() and .strip()
#   read as lines, tokenize each sentence
# pre-processing (optional)
#   replace infrequent tokens ( < 3? ) w/ <UNK>
# track counts/probabilities
#    use Python Counter or defaultdict
from perplexity_file import *
import numpy as np
import collections as cl


if __name__ == "__main__":

    train_file: str = "./data/1b_benchmark.train.tokens"
    processed_train_file: str = "./data/processed.1b_benchmark.train.tokens"
    dev_file: str = "./data/1b_benchmark.dev.tokens"
    test_file: str = "./data/1b_benchmark.test.tokens"

    print("Training uni-gram model")
    uni: NGramLM = NGramLM()
    uni.train(train_file) # will pre-process training file -> preprocessed_train_file
    print("Uni-Gram train PP:", uni.perplexity_file(processed_train_file))
    print("Uni-gram dev pp:", uni.perplexity_file(dev_file, unkify_OOV_words=True))
    # print("Uni-gram test pp:", uni.perplexity_file(test_file, unkify_OOV_words=True))

    print("Training bi-gram model")
    bi: NGramLM = NGramLM(n_grams=2)
    bi.train(processed_train_file, needs_preprocess=False)
    print("Bi-Gram train PP:", bi.perplexity_file(processed_train_file))
    print("Bi-gram dev pp:", bi.perplexity_file(dev_file, unkify_OOV_words=True))
    # print("Bi-gram test pp:", bi.perplexity_file(test_file, unkify_OOV_words=True))

    print("Training tri-gram model")
    tri: NGramLM = NGramLM(n_grams=3)
    tri.train(processed_train_file, needs_preprocess=False)
    print("Tri-Gram train PP:", tri.perplexity_file(processed_train_file))
    print("Tri-gram dev pp:", tri.perplexity_file(dev_file, unkify_OOV_words=True))
    # print("Tri-gram test pp:", tri.perplexity_file(test_file, unkify_OOV_words=True))



# Dictionary takes in ?ngram? as key, and returns count of ?ngram?
c = cl.Counter(a=4, b=2, c=0, d=-2)
sorted(c.elements())