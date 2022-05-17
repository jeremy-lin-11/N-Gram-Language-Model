# TODO
# Reorganize code
# move perplexity calculation into different file
# clean up train and perplexity calculations
# fix low Bigram train pp
# fix trigram hdtv pp being 1, and all other pp are pretty low as well
from concurrent.futures import process
from typing import Counter
# from perplexity import *
from train import *
import collections as cl


def main():

    train_file: str = "./data/1b_benchmark.train.tokens"
    processed_train_file: str = "./data/processed.1b_benchmark.train.tokens"
    dev_file: str = "./data/1b_benchmark.dev.tokens"
    test_file: str = "./data/1b_benchmark.test.tokens"
    testing_file: str = "./data/testing.tokens"
    sample: str = "./data/sample.tokens"

    print("\n PART 1 - N-grams w/out smoothing")

    print("\n   Training uni-gram model")
    uni: NGramLM = NGramLM(n_grams=1)
    uni.train(train_file, needs_preprocess=True) # will pre-process training file -> preprocessed_train_file 

    print("Uni-Gram HDTV PP:", uni.perplexity(testing_file))
    print("Uni-Gram train PP:", uni.perplexity(processed_train_file))
    print("Uni-gram dev pp:", uni.perplexity(dev_file))
    print("Uni-gram test pp:", uni.perplexity(test_file))

    print("\n   Training bi-gram model")
    bi: NGramLM = NGramLM(n_grams=2)
    bi.train(processed_train_file, needs_preprocess=False)
    print("Bi-Gram HDTV PP:", bi.perplexity(testing_file))
    print("Bi-Gram train PP:", bi.perplexity(processed_train_file))
    print("Bi-gram dev pp:", bi.perplexity(dev_file))
    print("Bi-gram test pp:", bi.perplexity(test_file))

    print("\n   Training tri-gram model")
    tri: NGramLM = NGramLM(n_grams=3)
    tri.train(processed_train_file, needs_preprocess=False)
    print("Tri-Gram HDTV PP:", tri.perplexity(testing_file))
    print("Tri-Gram train PP:", tri.perplexity(processed_train_file))
    print("Tri-gram dev pp:", tri.perplexity(dev_file))
    print("Tri-gram test pp:", tri.perplexity(test_file))


    print("\n PART 2 - Additive smoothing")
    
    print("\n   alpha = 1 (required)")
    alpha = 1
    print("\n       Training uni-gram model")
    uni: NGramLM = NGramLM(n_grams=1)
    uni.train(train_file, needs_preprocess=True) # will pre-process training file -> preprocessed_train_file 
    print("Uni-Gram HDTV PP:", uni.perplexity(testing_file, alpha))
    print("Uni-Gram train PP:", uni.perplexity(processed_train_file, alpha))
    print("Uni-gram dev pp:", uni.perplexity(dev_file, alpha))
    print("Uni-gram test pp:", uni.perplexity(test_file, alpha))

    print("\n       Training bi-gram model")
    bi: NGramLM = NGramLM(n_grams=2)
    bi.train(processed_train_file, needs_preprocess=False)
    print("Bi-Gram HDTV PP:", bi.perplexity(testing_file, alpha))
    print("Bi-Gram train PP:", bi.perplexity(processed_train_file, alpha))
    print("Bi-gram dev pp:", bi.perplexity(dev_file, alpha))
    print("Bi-gram test pp:", bi.perplexity(test_file, alpha))

    print("\n       Training tri-gram model")
    tri: NGramLM = NGramLM(n_grams=3)
    tri.train(processed_train_file, needs_preprocess=False)
    print("Tri-Gram HDTV PP:", tri.perplexity(testing_file, alpha))
    print("Tri-Gram train PP:", tri.perplexity(processed_train_file, alpha))
    print("Tri-gram dev pp:", tri.perplexity(dev_file, alpha))
    print("Tri-gram test pp:", tri.perplexity(test_file, alpha))


    print("\n   alpha = 2 (choice)")
    alpha = 2
    print("\n       Training uni-gram model")
    uni: NGramLM = NGramLM(n_grams=1)
    uni.train(train_file, needs_preprocess=True) # will pre-process training file -> preprocessed_train_file 
    print("Uni-Gram HDTV PP:", uni.perplexity(testing_file, alpha))
    print("Uni-Gram train PP:", uni.perplexity(processed_train_file, alpha))
    print("Uni-gram dev pp:", uni.perplexity(dev_file, alpha))
    print("Uni-gram test pp:", uni.perplexity(test_file, alpha))

    print("\n       Training bi-gram model")
    bi: NGramLM = NGramLM(n_grams=2)
    bi.train(processed_train_file, needs_preprocess=False)
    print("Bi-Gram HDTV PP:", bi.perplexity(testing_file, alpha))
    print("Bi-Gram train PP:", bi.perplexity(processed_train_file, alpha))
    print("Bi-gram dev pp:", bi.perplexity(dev_file, alpha))
    print("Bi-gram test pp:", bi.perplexity(test_file, alpha))

    print("\n       Training tri-gram model")
    tri: NGramLM = NGramLM(n_grams=3)
    tri.train(processed_train_file, needs_preprocess=False)
    print("Tri-Gram HDTV PP:", tri.perplexity(testing_file, alpha))
    print("Tri-Gram train PP:", tri.perplexity(processed_train_file, alpha))
    print("Tri-gram dev pp:", tri.perplexity(dev_file, alpha))
    print("Tri-gram test pp:", tri.perplexity(test_file, alpha))
    print("\n PART 3 - Smoothing w/ linear interpolation")

    print("\n   alpha = 5 (choice)")
    alpha = 5
    print("\n       Training uni-gram model")
    uni: NGramLM = NGramLM(n_grams=1)
    uni.train(train_file, needs_preprocess=True) # will pre-process training file -> preprocessed_train_file 
    print("Uni-Gram HDTV PP:", uni.perplexity(testing_file, alpha))
    print("Uni-Gram train PP:", uni.perplexity(processed_train_file, alpha))
    print("Uni-gram dev pp:", uni.perplexity(dev_file, alpha))
    print("Uni-gram test pp:", uni.perplexity(test_file, alpha))

    print("\n       Training bi-gram model")
    bi: NGramLM = NGramLM(n_grams=2)
    bi.train(processed_train_file, needs_preprocess=False)
    print("Bi-Gram HDTV PP:", bi.perplexity(testing_file, alpha))
    print("Bi-Gram train PP:", bi.perplexity(processed_train_file, alpha))
    print("Bi-gram dev pp:", bi.perplexity(dev_file, alpha))
    print("Bi-gram test pp:", bi.perplexity(test_file, alpha))

    print("\n       Training tri-gram model")
    tri: NGramLM = NGramLM(n_grams=3)
    tri.train(processed_train_file, needs_preprocess=False)
    print("Tri-Gram HDTV PP:", tri.perplexity(testing_file, alpha))
    print("Tri-Gram train PP:", tri.perplexity(processed_train_file, alpha))
    print("Tri-gram dev pp:", tri.perplexity(dev_file, alpha))
    print("Tri-gram test pp:", tri.perplexity(test_file, alpha))

    print("\n   TEST: Linear Interpolation")
    test: NGramLM = NGramLM(n_grams=3)
    test.train(processed_train_file, needs_preprocess=False) # will pre-process training file -> preprocessed_train_file 
    print("Interpolation HDTV PP:", test.interpolation(testing_file))
    print("Interpolation train PP:", test.interpolation(processed_train_file))
    print("Interpolation dev pp:", test.interpolation(dev_file))
    print("Interpolation test pp:", test.interpolation(test_file))

if __name__ == "__main__":
    main()



# Dictionary takes in ?ngram? as key, and returns count of ?ngram?