from collections import Counter, defaultdict
import numpy as np


class NGramLM(object):
    def __init__(self, n_grams=1):
        self.n_grams = n_grams
        self.origtypes = Counter()
        self.unigram = Counter()
        self.bigram = Counter()
        self.trigram = Counter()
        self.stop_token = '<STOP>'
        self.unk_token = '<UNK>'

    def trigramExtract(self, data):
        for sentence in range(0, len(data)):
            for word in range(0, len(data[sentence]) - 1):

                # if we reached the last word
                if word == len(data[sentence]) - 2:
                    word1 = data[sentence][word]
                    word2 = data[sentence][word+1]
                    word3 = self.stop_token
                    self.trigram[(word1, word2, word3)] += 1
                else:
                    word1 = data[sentence][word]
                    word2 = data[sentence][word + 1]
                    word3 = data[sentence][word + 2]
                    self.trigram[(word1, word2, word3)] += 1

        # print(self.trigram)

    def bigramExtract(self, data):
        for sentence in range(0, len(data)):
            for word in range(0, len(data[sentence])):
                if word == 0:
                    currWord = data[sentence][word]
                    prevWord = '<START>'
                    self.bigram[(currWord, prevWord)] += 1
                # if we reached the last word
                if word == len(data[sentence]) - 1:
                    prevWord = data[sentence][word]
                    currWord = self.stop_token
                    self.bigram[(currWord, prevWord)] += 1
                else:
                    prevWord = data[sentence][word]
                    currWord = data[sentence][word+1]
                    self.bigram[(currWord, prevWord)] += 1
        # print(self.bigram)

    def unigramExtract(self, data):
        for sentence in range(0, len(data)):
            for word in range(0, len(data[sentence])):
                self.unigram[data[sentence][word]] += 1
                if word == len(data[sentence]) - 1:
                    self.unigram[self.stop_token] += 1
        # print(self.unigram)

    def train(self, file_path, needs_preprocess=True):

        #TOKENIZING
        with open(file_path, 'r', encoding='utf-8') as file:
            data = file.readlines()
            for sentence in range(0, len(data)):
                data[sentence] = data[sentence].strip(' \n').split(' ') 
                # print("data[sentence] =", data[sentence])
                for word in data[sentence]:
                    # print("self.origtypes =", self.origtypes)
                    # print("data[sentence][word] =", data[sentence][word])
                    self.origtypes[word] += 1
        # print(self.origtypes)
        data = list(filter((['']).__ne__, data))    #fixes empty new lines being stored         
        # print(data)


        #PREPROCESSING (OPTIONAL)
        #need to unkify words (OOV if less than 3 counts)
        if(needs_preprocess):
            with open("./data/processed.1b_benchmark.train.tokens", 'w', encoding='utf-8') as file:
                for sentence in range(0, len(data)):
                    for word in range(0, len(data[sentence])):
                        # print("origtypes for word '", data[sentence][word], "' is", self.origtypes[data[sentence][word]])
                        if self.origtypes[data[sentence][word]] >= 3:
                            file.write(data[sentence][word])
                        else:
                            file.write(self.unk_token)

                        if word != len(data[sentence]) - 1:
                            file.write(' ')
                        else:
                            file.write('\n')

            with open("./data/processed.1b_benchmark.train.tokens", 'r', encoding='utf-8') as file:
                processedData = file.readlines()
                for sentence in range(0, len(processedData)):
                    processedData[sentence] = processedData[sentence].strip(' \n').split(' ')
            data = processedData
        # print(data)
        # print("done and n_grams is:", self.n_grams)

        #TRACK COUNTS/PROBABILITIES 
        #use counter dictionary, ngram key, count value

        # UNIGRAM EXTRACTION
        if (self.n_grams == 1):
            self.unigramExtract(data)
            print("unigram counter length = ", len(self.unigram))
            # print(self.unigram)

        # BIGRAM EXTRACTION
        elif (self.n_grams == 2):
            self.unigramExtract(data)
            self.bigramExtract(data)
            print('unigram', self.unigram)
            print('bigram', self.bigram)

        # TRIGRAM EXTRACTION
        elif (self.n_grams == 3):
            self.unigramExtract(data)
            self.bigramExtract(data)
            self.trigramExtract(data)
            
        return 
    
    # tokenization, compute sentence probabilities
    #   unkify_OOV_words: the purpose of <UNK> is to generalize 
    #   to infrequent words in your dev/test set: if not in 
    #   the 26602 vocab words, replace w/ UNK before computing probability

    # this thing should take the ngram model we made from train and use those counts in calculating the MLE for test/dev sets
    # tokenize test set, counts and all that of ngrams, use perplexity
    # but if we're comparing test p(xi) in perplexity, 
    # and word shows up we havent seen before? it would *0 or log0 and fuck up the perplexity formula?
    def perplexity(self, file_path):
        #TOKENIZING
        with open(file_path, 'r', encoding='utf-8') as file:
            testingData = file.readlines()
            for sentence in range(0, len(testingData)):
                testingData[sentence] = testingData[sentence].strip(' \n').split(' ') 
                # print("data[sentence] =", data[sentence])
                for word in testingData[sentence]:
                    # print("self.origtypes =", self.origtypes)
                    # print("data[sentence][word] =", data[sentence][word])
                    self.origtypes[word] += 1
        # print(self.origtypes)
        testingData = list(filter((['']).__ne__, testingData))    #fixes empty new lines being stored         
        # print(testingData)

        #Perplexity Calculation for Unigrams
        if self.n_grams == 1:
            print("unigram", self.unigram)
            prSum = 0
            N = self.unigram.total()
            M=0
            print("N=", N)
            for sentence in range(0,len(testingData)):
                M += len(testingData[sentence]) + 1
                #calculate pr of each word
                prSentence = 0
                for word in range(0,len(testingData[sentence])):
                    if self.unigram[testingData[sentence][word]] == 0:
                        prSentence += self.unigram[self.unk_token] / N
                    else:
                        prSentence += self.unigram[testingData[sentence][word]] / N
                prSentence += self.unigram[self.stop_token] / N
                prSum += np.log2(prSentence)

            print("M=", M)
            print("sum" , prSum)
            L = prSum / M
            print("L", L)
            perplexity = np.power(2, L)
            return perplexity

        elif self.n_grams == 2:
            prSum = 0
            M=0
            for sentence in range(0,len(testingData)):
                M += len(testingData[sentence]) + 1
                #calculate pr of each word
                prSentence = 0
                for word in range(0,len(testingData[sentence])):
                    if word == 0:
                        currWord = testingData[sentence][word]
                        prSentence += self.bigram[(currWord, '<START>')] / self.unigram[self.stop_token]
                    if word == len(testingData[sentence]) - 1:
                        prevWord = testingData[sentence][word]
                        prSentence += self.bigram[('<STOP>', prevWord)] / self.unigram[prevWord]
                    else:
                        currWord = testingData[sentence][word+1]
                        prevWord = testingData[sentence][word]
                        prSentence += self.bigram[(currWord, prevWord)] / self.unigram[prevWord]          

                prSum += np.log2(prSentence)

            print("M=", M)
            print("sum" , prSum)
            L = prSum / M
            print("L", L)
            perplexity = np.power(2, L)
            return perplexity



