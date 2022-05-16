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
            for word in range(0, len(data[sentence])):
                # first word, two starts
                if word == 0 and len(data[sentence]) != 1:
                    currWord = data[sentence][word]
                    prevWord1 = '<START>'
                    prevWord2 = '<START>'
                    # print('index', word, 'gives', '1')
                    self.trigram[(prevWord2, prevWord1, currWord)] += 1
                #second word, one start
                elif word == 1 and len(data[sentence]) != 1:
                    currWord = data[sentence][word]
                    prevWord1 = data[sentence][word-1]
                    prevWord2 = '<START>'
                    self.trigram[(prevWord2, prevWord1, currWord)] += 1
                    # print('index', word, 'gives', '2')
                    if word == len(data[sentence]) - 1:
                        prevWord2 = data[sentence][word-1]
                        prevWord1 = data[sentence][word]
                        currWord = self.stop_token
                        # print('index', word, 'gives', '3')
                        self.trigram[(prevWord2, prevWord1, currWord)] += 1
                #edge case 1 token sentence
                elif word == 0 and len(data[sentence]) == 1:
                    currWord = data[sentence][word]
                    prevWord1 = '<START>'
                    prevWord2 = '<START>'
                    # print('index', word, 'gives', '1')
                    self.trigram[(prevWord2, prevWord1, currWord)] += 1

                    currWord = '<STOP>'
                    prevWord1 = data[sentence][word]
                    prevWord2 = '<START>'
                    self.trigram[(prevWord2, prevWord1, currWord)] += 1
                    # print('index', word, 'gives', '2')
                # if we reached the last word, stop token
                elif word == len(data[sentence]) - 1 and len(data[sentence]) != 1:
                    prevWord2 = data[sentence][word-1]
                    prevWord1 = data[sentence][word]
                    currWord = self.stop_token
                    # print('index', word, 'gives', '3')
                    self.trigram[(prevWord2, prevWord1, currWord)] += 1
                # for all else
                elif len(data[sentence]) != 1:
                    prevWord2 = data[sentence][word-2]
                    prevWord1 = data[sentence][word-1]
                    currWord = data[sentence][word]
                    # print('index', word, 'gives', '4')
                    self.trigram[(prevWord2, prevWord1, currWord)] += 1

        # print(self.trigram)

    def bigramExtract(self, data):
        for sentence in range(0, len(data)):
            for word in range(0, len(data[sentence])):
                #if we are at the start
                if word == 0 and len(data[sentence]) != 1:
                    currWord = data[sentence][word]
                    prevWord = '<START>'
                    self.bigram[(prevWord, currWord)] += 1
                # if we reached the last word
                elif word == len(data[sentence]) - 1 and len(data[sentence]) != 1:
                    currWord = data[sentence][word]
                    prevWord = data[sentence][word-1]
                    self.bigram[(prevWord, currWord)] += 1

                    currWord = '<STOP>'
                    prevWord = data[sentence][word]
                    self.bigram[(prevWord, currWord)] += 1
                elif len(data[sentence]) != 1:
                    currWord = data[sentence][word]
                    prevWord = data[sentence][word-1]
                    self.bigram[(prevWord, currWord)] += 1
                # edge case 1 token sentence
                elif len(data[sentence]) == 1:
                    currWord = data[sentence][word]
                    prevWord = '<START>'
                    self.bigram[(prevWord, currWord)] += 1

                    currWord = '<STOP>'
                    prevWord = data[sentence][word]
                    self.bigram[(prevWord, currWord)] += 1
                
        # print(self.bigram)

    def unigramExtract(self, data):
        for sentence in range(0, len(data)):
            for word in range(0, len(data[sentence])):
                self.unigram[data[sentence][word]] += 1
                if word == len(data[sentence]) - 1:
                    self.unigram[self.stop_token] += 1
        # print(self.unigram)

    def train(self, file_path, needs_preprocess=True):
        wordCount = 0
        #TOKENIZING
        with open(file_path, 'r', encoding='utf-8') as file:
            data = file.readlines()
            for sentence in range(0, len(data)):
                data[sentence] = data[sentence].strip(' \n').split(' ') 
                # print("data[sentence] =", data[sentence])
                for word in data[sentence]:
                    # print("self.origtypes =", self.origtypes)
                    # print("data[sentence][word] =", data[sentence][word])
                    wordCount += 1
                    self.origtypes[word] += 1
        # print(self.origtypes)
        data = list(filter((['']).__ne__, data))    #fixes empty new lines being stored         
        # print(data)
        print('WORDCOUNT', wordCount)

        #PREPROCESSING (OPTIONAL)
        #need to unkify words (OOV if less than 3 counts)
        if(needs_preprocess):
            wordCount = 0
            with open("./data/processed.1b_benchmark.train.tokens", 'w', encoding='utf-8') as file:
                for sentence in range(0, len(data)):
                    for word in range(0, len(data[sentence])):
                        # print("origtypes for word '", data[sentence][word], "' is", self.origtypes[data[sentence][word]])
                        if self.origtypes[data[sentence][word]] >= 3:
                            file.write(data[sentence][word])
                        else:
                            file.write('<UNK>')

                        if word != len(data[sentence]) - 1:
                            file.write(' ')
                        else:
                            file.write('\n')

            with open("./data/processed.1b_benchmark.train.tokens", 'r', encoding='utf-8') as file:
                processedData = file.readlines()
                for sentence in range(0, len(processedData)):
                    processedData[sentence] = processedData[sentence].strip(' \n').split(' ')
                    for word in range(0,len(processedData[sentence])):
                        wordCount += 1
            data = processedData
            print('processed wordcount', wordCount)
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
            print("unigram counter length = ", len(self.unigram))
            print("bigram counter length = ", len(self.bigram))
            # print('unigram', self.unigram)
            # print('bigram', self.bigram)

        # TRIGRAM EXTRACTION
        elif (self.n_grams == 3):
            self.unigramExtract(data)
            self.bigramExtract(data)
            self.trigramExtract(data)
            print("unigram counter length = ", len(self.unigram))
            print("bigram counter length = ", len(self.bigram))
            print("trigram counter length = ", len(self.trigram))
            # print('unigram', self.unigram)
            # print('bigram', self.bigram)
            # print('trigram', self.trigram)
            
        return 

    def interpolation(self, file_path):
        l1 = 0.1
        l2 = 0.3
        l3 = 0.6
        #TOKENIZING
        unigramSize = self.unigram.total()
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

        # print(self.unigram['<UNK>'])
        prSum = 0
        testTotalTokens=0
        for sentence in range(0,len(testingData)):
            testTotalTokens += len(testingData[sentence]) + 1
            prSentence = 0
            for word in range(0,len(testingData[sentence])):
                if word == 0 and len(testingData[sentence]) != 1:
                    if self.unigram[testingData[sentence][word]] == 0:
                        # currWord = '<UNK>'
                        continue
                    else: currWord = testingData[sentence][word]
                    prevWord1 = '<START>'
                    prevWord2 = '<START>'
                    if self.trigram[('<START>', '<START>', currWord)] != 0:
                        # bigram denominator for first start token is start start, equiv to # of stop tokens
                        term1 = l1 * (np.log2(self.unigram[currWord]) - np.log2(unigramSize))
                        term2 = l2 * (np.log2(self.bigram[(prevWord1, currWord)]) - np.log2(self.unigram['<STOP>']))
                        term3 = l3 * (np.log2(self.trigram[(prevWord2, prevWord1, currWord)]) - np.log2(self.unigram['<STOP>']))
                        prSentence += term1 + term2 + term3
                elif word == 1 and len(testingData[sentence]) != 1:
                    if self.unigram[testingData[sentence][word]] == 0:
                        # currWord = '<UNK>'
                        continue
                    else: currWord = testingData[sentence][word]
                    if self.unigram[testingData[sentence][word-1]] == 0:
                        # prevWord1 = '<UNK>'
                        continue
                    else: prevWord1 = testingData[sentence][word-1]
                    prevWord2 = '<START>'

                    if self.trigram[('<START>', prevWord1, currWord)] != 0 and self.bigram[('<START>', prevWord1)] != 0:
                        term1 = l1 * (np.log2(self.unigram[currWord]) - np.log2(unigramSize))
                        term2 = l2 * (np.log2(self.bigram[(prevWord1, currWord)]) - np.log2(self.unigram[prevWord1]))
                        term3 = l3 * (np.log2(self.trigram[(prevWord2, prevWord1, currWord)]) - np.log2(self.bigram[(prevWord2, prevWord1)]))
                        prSentence += term1 + term2 + term3
                    
                    if word == len(testingData[sentence]) - 1:
                        if self.unigram[testingData[sentence][word-1]] == 0:
                            # prevWord2 = '<UNK>'
                            continue
                        else: prevWord2 = testingData[sentence][word-1]
                        if self.unigram[testingData[sentence][word]] == 0:
                            # prevWord1 = '<UNK>'
                            continue
                        else: prevWord1 = testingData[sentence][word]
                        currWord = '<STOP>'

                        if self.trigram[(prevWord2, prevWord1, currWord)] != 0 and self.bigram[(prevWord2, prevWord1)] != 0:
                            term1 = l1 * (np.log2(self.unigram[currWord]) - np.log2(unigramSize))
                            term2 = l2 * (np.log2(self.bigram[(prevWord1, currWord)]) - np.log2(self.unigram[prevWord1]))
                            term3 = l3 * (np.log2(self.trigram[(prevWord2, prevWord1, currWord)]) - np.log2(self.bigram[(prevWord2, prevWord1)]))
                            prSentence += term1 + term2 + term3


                elif word == 0 and len(testingData[sentence]) == 1:
                    # edge case for single token sentence
                    # calculate for first word and for stop token
                    if self.unigram[testingData[sentence][word]] == 0: 
                        # currWord = '<UNK>'
                        continue
                    else: currWord = testingData[sentence][word]
                    prevWord1 = '<START>'
                    prevWord2 = '<START>'
                    if self.trigram[('<START>', '<START>', currWord)] != 0 and self.bigram[('<START>', currWord)] != 0:
                        term1 = l1 * (np.log2(self.unigram[currWord]) - np.log2(unigramSize))
                        term2 = l2 * (np.log2(self.bigram[(prevWord1, currWord)]) - np.log2(self.unigram['<STOP>']))
                        term3 = l3 * (np.log2(self.trigram[(prevWord2, prevWord1, currWord)]) - np.log2(self.unigram['<STOP>']))
                        prSentence += term1 + term2 + term3

                    currWord = '<STOP>'
                    if self.unigram[testingData[sentence][word]] == 0:
                        # prevWord1 = '<UNK>'
                        continue
                    else: prevWord1 = testingData[sentence][word]
                    prevWord2 = '<START>'
                    if self.trigram[('<START>', prevWord1, currWord)] != 0 and self.bigram[('<START>', prevWord1)] != 0:
                        term1 = l1 * (np.log2(self.unigram[currWord]) - np.log2(unigramSize))
                        term2 = l2 * (np.log2(self.bigram[(prevWord1, currWord)]) - np.log2(self.unigram[prevWord1]))
                        term3 = l3 * (np.log2(self.trigram[(prevWord2, prevWord1, currWord)]) - np.log2(self.bigram[(prevWord2, prevWord1)]))
                        prSentence += term1 + term2 + term3
                        
                elif word == len(testingData[sentence]) - 1:
                    # if at last word, calculate for stop token
                    currWord = '<STOP>'
                    if self.unigram[testingData[sentence][word]] == 0:
                        # prevWord1 = '<UNK>'
                        continue
                    else: prevWord1 = testingData[sentence][word]

                    if self.unigram[testingData[sentence][word-1]] == 0:
                        # prevWord2 = '<UNK>'
                        continue
                    else: prevWord2 = testingData[sentence][word-1]
                    if self.trigram[(prevWord2, prevWord1, '<STOP>')] != 0 and self.bigram[(prevWord2, prevWord1)] != 0:
                        term1 = l1 * (np.log2(self.unigram[currWord]) - np.log2(unigramSize))
                        term2 = l2 * (np.log2(self.bigram[(prevWord1, currWord)]) - np.log2(self.unigram[prevWord1]))
                        term3 = l3 * (np.log2(self.trigram[(prevWord2, prevWord1, currWord)]) - np.log2(self.bigram[(prevWord2, prevWord1)]))
                        prSentence += term1 + term2 + term3
                elif len(testingData[sentence]) != 1:
                    if self.unigram[testingData[sentence][word]] == 0:
                        # currWord = '<UNK>'
                        continue
                    else: currWord = testingData[sentence][word]

                    if self.unigram[testingData[sentence][word-1]] == 0:
                        # prevWord1 = '<UNK>'
                        continue
                    else:  prevWord1 = testingData[sentence][word-1]

                    if self.unigram[testingData[sentence][word-2]] == 0:
                        # prevWord2 = '<UNK>'
                        continue
                    else: prevWord2 = testingData[sentence][word-2]
                    if self.trigram[(prevWord2, prevWord1, currWord)] != 0 and self.bigram[(prevWord2, prevWord1)] != 0:
                        term1 = l1 * (np.log2(self.unigram[currWord]) - np.log2(unigramSize))
                        term2 = l2 * (np.log2(self.bigram[(prevWord1, currWord)]) - np.log2(self.unigram[prevWord1]))
                        term3 = l3 * (np.log2(self.trigram[(prevWord2, prevWord1, currWord)]) - np.log2(self.bigram[(prevWord2, prevWord1)]))
                        prSentence += term1 + term2 + term3        

            prSum += prSentence

        # print("testTotalTokens=", testTotalTokens)
        # print("sum" , prSum)
        L = prSum / testTotalTokens
        # print("L", L)
        perplexity = np.power(2, -1 * L)
        return perplexity
            
    
    # tokenization, compute sentence probabilities
    #   unkify_OOV_words: the purpose of <UNK> is to generalize 
    #   to infrequent words in your dev/test set: if not in 
    #   the 26602 vocab words, replace w/ UNK before computing probability

    # this thing should take the ngram model we made from train and use those counts in calculating the MLE for test/dev sets
    # tokenize test set, counts and all that of ngrams, use perplexity
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
        
        alpha = 0
        vocabSize = len(self.unigram)
        

        #UNIGRAM PERPLEXITY
        if self.n_grams == 1:
            # print("unigram", self.unigram)
            prSum = 0
            unigramSize = self.unigram.total() #sum of tokens in vocabulary
            testTotalTokens=0 #total tokens in all sentences
            # print("unigramSize=", unigramSize)
            # print(self.unigram['<UNK>'])

            #for each sentence
            for sentence in range(0,len(testingData)):
                testTotalTokens += len(testingData[sentence]) + 1 #token in sentence + stop
                #calculate pr of each word in sentence, add to summation
                prSentence = 0
                for word in range(0,len(testingData[sentence])):
                    if self.unigram[testingData[sentence][word]] == 0:
                        prSentence += np.log2(self.unigram['<UNK>'] + alpha) - np.log2(unigramSize + alpha * vocabSize)
                        # print('word pr' , self.unigram[self.unk_token] / unigramSize)
                        # print('sentence pr', prSentence)
                    else:
                        prSentence += np.log2(self.unigram[testingData[sentence][word]] + alpha) -  np.log2(unigramSize + alpha * vocabSize)
                        # print('word pr' , self.unigram[testingData[sentence][word]] / unigramSize)
                        # print('sentence pr', prSentence)
                prSentence += np.log2(self.unigram['<STOP>'] + alpha) - np.log2(unigramSize + alpha * vocabSize)
                
                # print('word pr' , self.unigram[self.stop_token] / unigramSize)
                # print('sentence pr', prSentence)
                prSum += prSentence


            # print("testTotalTokens=", testTotalTokens)
            # print("sum" , prSum)
            L = prSum / testTotalTokens
            # print("L", L)
            perplexity = np.power(2, -1 * L)
            return perplexity

        #BIGRAM PERPLEXITY
        elif self.n_grams == 2:
            # print(self.unigram['<UNK>'])
            prSum = 0
            testTotalTokens=0
            for sentence in range(0,len(testingData)):
                testTotalTokens += len(testingData[sentence]) + 1
                #calculate pr of each word
                prSentence = 0
                for word in range(0,len(testingData[sentence])):
                    # if we are at the start
                    if word == 0 and len(testingData[sentence]) != 1:
                        if self.unigram[testingData[sentence][word]] == 0:
                            # currWord = '<UNK>'
                            continue
                        else:
                            currWord = testingData[sentence][word]

                        if self.bigram[('<START>', currWord)] != 0:
                            prSentence += np.log2(self.bigram[('<START>', currWord)] + alpha) - np.log2(self.unigram[self.stop_token] + alpha * vocabSize)
                    
                    # if we reached the last word
                    elif word == len(testingData[sentence]) - 1 and len(testingData[sentence]) != 1:
                        #calculate for word at index
                        if self.unigram[testingData[sentence][word]] == 0:
                            # currWord = '<UNK>'
                            continue
                        else:
                            currWord = testingData[sentence][word]
                        if self.unigram[testingData[sentence][word-1]] == 0:
                            # prevWord = '<UNK>'
                            continue
                        else:
                            prevWord = testingData[sentence][word-1]
                            
                        if self.bigram[(prevWord, currWord)] != 0 and self.unigram[prevWord] != 0:
                            prSentence += np.log2(self.bigram[(prevWord, currWord)] + alpha) - np.log2(self.unigram[prevWord] + alpha * vocabSize)

                        # calculate for <STOP>
                        if self.unigram[testingData[sentence][word]] == 0:
                            # prevWord = '<UNK>'
                            continue
                        else:
                            prevWord = testingData[sentence][word]

                        if self.bigram[(prevWord, '<STOP>')] != 0 and self.unigram[prevWord] != 0:
                            prSentence += np.log2(self.bigram[(prevWord, '<STOP>')] + alpha) - np.log2(self.unigram[prevWord] + alpha * vocabSize)
                    elif len(testingData[sentence]) != 1:
                        if self.unigram[testingData[sentence][word]] == 0:
                            # currWord = '<UNK>'
                            continue
                        else:
                            currWord = testingData[sentence][word]
                        if self.unigram[testingData[sentence][word-1]] == 0:
                            # prevWord = '<UNK>'
                            continue
                        else:
                            prevWord = testingData[sentence][word-1]
                        
                        if self.bigram[(prevWord, currWord)] != 0 and self.unigram[prevWord] != 0:
                            prSentence += np.log2(self.bigram[(prevWord, currWord)] + alpha) - np.log2(self.unigram[prevWord] + alpha * vocabSize)   
                    # edge case for 1 token sentence 
                    elif len(testingData[sentence]) == 1:   
                        # add for word at index
                        if self.unigram[testingData[sentence][word]] == 0:
                            # currWord = '<UNK>'
                            continue
                        else:
                            currWord = testingData[sentence][word]
                        if self.bigram[('<START>', currWord)] != 0:
                            prSentence += np.log2(self.bigram[('<START>', currWord)] + alpha) - np.log2(self.unigram[self.stop_token] + alpha * vocabSize)

                        # add for stop
                        currWord = '<STOP>'
                        prevWord = currWord = testingData[sentence][word]
                        if self.bigram[('<START>', currWord)] != 0:
                            prSentence += np.log2(self.bigram[('<START>', currWord)] + alpha) - np.log2(self.unigram[self.stop_token] + alpha * vocabSize)


                prSum += prSentence

            # print("testTotalTokens=", testTotalTokens)
            # print("sum" , prSum)
            L = prSum / testTotalTokens
            # print("L", L)
            perplexity = np.power(2, -1 * L)
            return perplexity
        
        #TRIGRAM PERPLEXITY
        elif self.n_grams == 3:
            # print(self.unigram['<UNK>'])
            prSum = 0
            testTotalTokens=0
            for sentence in range(0,len(testingData)):
                testTotalTokens += len(testingData[sentence]) + 1
                # if len(testingData[sentence]) <= 1:
                #     print(sentence)
                #calculate pr of each word
                prSentence = 0
                for word in range(0,len(testingData[sentence])):
                    if word == 0 and len(testingData[sentence]) != 1:
                        # if at first word
                        # calculate for both word at index and word at index+1

                        if self.unigram[testingData[sentence][word]] == 0:
                            # currWord = '<UNK>'
                            continue
                        else: currWord = testingData[sentence][word]
                        prevWord1 = '<START>'
                        prevWord2 = '<START>'
                        # print('prevWord1: word at index ', word, 'is: ', testingData[sentence][word])
                        # print('currWord: word at index ', word, '+ 1 is: ', testingData[sentence][word+1])
                        # print('currWord is: ', currWord, 'prevWord1 is: ', '<START>', 'prevWord2 is: ', '<PRESTART>')
                        # print('trigram count', self.trigram[('<PRESTART>', '<START>', currWord)])
                        # print('bigram count', self.bigram[('<START>', currWord)])
                        if self.trigram[('<START>', '<START>', currWord)] != 0:
                            # bigram denominator for first start token is start start, equiv to # of stop tokens
                            prSentence += np.log2(self.trigram[('<START>', '<START>', currWord)] + alpha) - np.log2(self.unigram['<STOP>'] + alpha * vocabSize)

                    elif word == 1 and len(testingData[sentence]) != 1:
                        if self.unigram[testingData[sentence][word]] == 0:
                            # currWord = '<UNK>'
                            continue
                        else: currWord = testingData[sentence][word]
                        if self.unigram[testingData[sentence][word-1]] == 0:
                            # prevWord1 = '<UNK>'
                            continue
                        else: prevWord1 = testingData[sentence][word-1]
                        prevWord2 = '<START>'
                        if self.trigram[('<START>', prevWord1, currWord)] != 0 and self.bigram[('<START>', prevWord1)] != 0:
                            prSentence += np.log2(self.trigram[('<START>', prevWord1, currWord)] + alpha) - np.log2(self.bigram[('<START>', prevWord1)] + alpha * vocabSize)
                        
                        if word == len(testingData[sentence]) - 1:
                            if self.unigram[testingData[sentence][word-1]] == 0:
                                # prevWord2 = '<UNK>'
                                continue
                            else: prevWord2 = testingData[sentence][word-1]
                            if self.unigram[testingData[sentence][word]] == 0:
                                # prevWord1 = '<UNK>'
                                continue
                            else: prevWord1 = testingData[sentence][word]
                            currWord = '<STOP>'

                            if self.trigram[(prevWord2, prevWord1, currWord)] != 0 and self.bigram[(prevWord2, prevWord1)] != 0:
                                prSentence += np.log2(self.trigram[(prevWord2, prevWord1, currWord)] + alpha) - np.log2(self.bigram[(prevWord2, prevWord1)] + alpha * vocabSize)


                    elif word == 0 and len(testingData[sentence]) == 1:
                        # edge case for single token sentence
                        # calculate for first word and for stop token
                        if self.unigram[testingData[sentence][word]] == 0: 
                            # currWord = '<UNK>'
                            continue
                        else: currWord = testingData[sentence][word]
                        prevWord1 = '<START>'
                        prevWord2 = '<START>'
                        # print('prevWord1: word at index ', word, 'is: ', testingData[sentence][word])
                        # print('currWord: word at index ', word, '+ 1 is: ', testingData[sentence][word+1])
                        # print('currWord is: ', currWord, 'prevWord1 is: ', '<START>', 'prevWord2 is: ', '<PRESTART>')
                        # print('trigram count', self.trigram[('<PRESTART>', '<START>', currWord)])
                        # print('bigram count', self.bigram[('<START>', currWord)])
                        if self.trigram[('<START>', '<START>', currWord)] != 0 and self.bigram[('<START>', currWord)] != 0:
                            prSentence += np.log2(self.trigram[('<START>', '<START>', currWord)] + alpha) - np.log2(self.bigram[('<START>', currWord)] + alpha * vocabSize)

                        currWord = '<STOP>'
                        if self.unigram[testingData[sentence][word]] == 0:
                            # prevWord1 = '<UNK>'
                            continue
                        else: prevWord1 = testingData[sentence][word]
                        prevWord2 = '<START>'
                        # print('prevWord1: word at index ', word, 'is: ', testingData[sentence][word])
                        # print('currWord: word at index ', word, '+ 1 is: ', testingData[sentence][word+1])
                        # print('currWord is: ', currWord, 'prevWord1 is: ', prevWord1, 'prevWord2 is: ', '<START>')
                        # print('trigram count', self.trigram[('<START>', prevWord1, currWord)])
                        # print('bigram count', self.bigram[('<START>', prevWord1)])
                        if self.trigram[('<START>', prevWord1, currWord)] != 0 and self.bigram[('<START>', prevWord1)] != 0:
                            prSentence += np.log2(self.trigram[('<START>', prevWord1, currWord)] + alpha) - np.log2(self.bigram[('<START>', prevWord1)] + alpha * vocabSize)
                            
                    elif word == len(testingData[sentence]) - 1:
                        # if at last word, calculate for stop token
                        # currWord = self.stop_token
                        if self.unigram[testingData[sentence][word]] == 0:
                            # prevWord1 = '<UNK>'
                            continue
                        else: prevWord1 = testingData[sentence][word]

                        if self.unigram[testingData[sentence][word-1]] == 0:
                            # prevWord2 = '<UNK>'
                            continue
                        else: prevWord2 = testingData[sentence][word-1]
                        # print('prevWord1: word at index ', word, 'is: ', testingData[sentence][word])
                        # print('prevWord2: word at index ', word, '- 1 is: ', testingData[sentence][word-1])
                        # print('currWord is: ', '<STOP>', 'prevWord1 is: ', prevWord1, 'prevWord2 is: ', prevWord2)
                        # print('trigram count', self.trigram[(prevWord2, prevWord1, '<STOP>')])
                        # print('bigram count', self.bigram[(prevWord2, prevWord1)])
                        if self.trigram[(prevWord2, prevWord1, '<STOP>')] != 0 and self.bigram[(prevWord2, prevWord1)] != 0:
                            prSentence += np.log2(self.trigram[(prevWord2, prevWord1, '<STOP>')] + alpha) - np.log2(self.bigram[(prevWord2, prevWord1)] + alpha * vocabSize)
                    elif len(testingData[sentence]) != 1:
                        # calculate for index+1
                        if self.unigram[testingData[sentence][word]] == 0:
                            # currWord = '<UNK>'
                            continue
                        else: currWord = testingData[sentence][word]

                        if self.unigram[testingData[sentence][word-1]] == 0:
                            # prevWord1 = '<UNK>'
                            continue
                        else:  prevWord1 = testingData[sentence][word-1]

                        if self.unigram[testingData[sentence][word-2]] == 0:
                            # prevWord2 = '<UNK>'
                            continue
                        else: prevWord2 = testingData[sentence][word-2]
                        
                        # print('prevWord2: word at index ', word, '- 1 is: ', testingData[sentence][word-1])
                        # print('prevWord1: word at index ', word, 'is: ', testingData[sentence][word])
                        # print('currWord: word at index ', word, '+ 1 is: ', testingData[sentence][word+1])
                        # print('currWord is: ', currWord, 'prevWord1 is: ', prevWord1, 'prevWord2 is: ', prevWord2)
                        # print('trigram count', self.trigram[(prevWord2, prevWord1, currWord)])
                        # print('bigram count', self.bigram[(prevWord2, prevWord1)])
                        if self.trigram[(prevWord2, prevWord1, currWord)] != 0 and self.bigram[(prevWord2, prevWord1)] != 0:
                            prSentence += np.log2(self.trigram[(prevWord2, prevWord1, currWord)] + alpha) - np.log2(self.bigram[(prevWord2, prevWord1)] + alpha * vocabSize)        

                prSum += prSentence

            # print("testTotalTokens=", testTotalTokens)
            # print("sum" , prSum)
            L = prSum / testTotalTokens
            # print("L", L)
            perplexity = np.power(2, -1 * L)
            return perplexity



