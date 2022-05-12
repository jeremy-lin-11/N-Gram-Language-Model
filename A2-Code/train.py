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
                    prevWord2 = '<PRESTART>'
                    # print('index', word, 'gives', '1')
                    self.trigram[(prevWord2, prevWord1, currWord)] += 1

                    currWord = data[sentence][word+1]
                    prevWord1 = data[sentence][word]
                    prevWord2 = '<START>'
                    self.trigram[(prevWord2, prevWord1, currWord)] += 1
                    # print('index', word, 'gives', '2')
                elif word == 0 and len(data[sentence]) == 1:
                    currWord = data[sentence][word]
                    prevWord1 = '<START>'
                    prevWord2 = '<PRESTART>'
                    # print('index', word, 'gives', '1')
                    self.trigram[(prevWord2, prevWord1, currWord)] += 1

                    currWord = '<STOP>'
                    prevWord1 = data[sentence][word]
                    prevWord2 = '<START>'
                    self.trigram[(prevWord2, prevWord1, currWord)] += 1
                    # print('index', word, 'gives', '2')
                # if we reached the last word
                elif word == len(data[sentence]) - 1 and len(data[sentence]) != 1:
                    prevWord2 = data[sentence][word-1]
                    prevWord1 = data[sentence][word]
                    currWord = self.stop_token
                    # print('index', word, 'gives', '3')
                    self.trigram[(prevWord2, prevWord1, currWord)] += 1
                elif len(data[sentence]) != 1:
                    prevWord2 = data[sentence][word-1]
                    prevWord1 = data[sentence][word]
                    currWord = data[sentence][word+1]
                    # print('index', word, 'gives', '4')
                    self.trigram[(prevWord2, prevWord1, currWord)] += 1

        # print(self.trigram)

    def bigramExtract(self, data):
        for sentence in range(0, len(data)):
            for word in range(0, len(data[sentence])):
                #if we are at the start
                if word == 0:
                    currWord = data[sentence][word]
                    prevWord = '<START>'
                    self.bigram[(prevWord, currWord)] += 1
                # if we reached the last word
                if word == len(data[sentence]) - 1:
                    prevWord = data[sentence][word]
                    currWord = self.stop_token
                    self.bigram[(prevWord, currWord)] += 1
                else:
                    prevWord = data[sentence][word]
                    currWord = data[sentence][word+1]
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
            # print("unigram counter length = ", len(self.unigram))
            # print(self.unigram)

        # BIGRAM EXTRACTION
        elif (self.n_grams == 2):
            self.unigramExtract(data)
            self.bigramExtract(data)
            # print("unigram counter length = ", len(self.unigram))
            # print("bigram counter length = ", len(self.bigram))
            # print('unigram', self.unigram)
            # print('bigram', self.bigram)

        # TRIGRAM EXTRACTION
        elif (self.n_grams == 3):
            self.unigramExtract(data)
            self.bigramExtract(data)
            self.trigramExtract(data)
            # print("unigram counter length = ", len(self.unigram))
            # print("bigram counter length = ", len(self.bigram))
            # print("trigram counter length = ", len(self.trigram))
            # print('unigram', self.unigram)
            # print('bigram', self.bigram)
            # print('trigram', self.trigram)
            
        return 
    
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

        #UNIGRAM PERPLEXITY
        if self.n_grams == 1:
            # print("unigram", self.unigram)
            prSum = 0
            unigramSize = self.unigram.total() #sum of tokens in vocabulary
            testTotalTokens=0 #total tokens in all sentences
            # print("unigramSize=", unigramSize)


            #for each sentence
            for sentence in range(0,len(testingData)):
                testTotalTokens += len(testingData[sentence]) + 1 #token in sentence + stop
                #calculate pr of each word in sentence, add to summation
                prSentence = 0
                for word in range(0,len(testingData[sentence])):
                    if self.unigram[testingData[sentence][word]] == 0:
                        prSentence += np.log2(self.unigram['<UNK>']) - np.log2(unigramSize)
                        # print('word pr' , self.unigram[self.unk_token] / unigramSize)
                        # print('sentence pr', prSentence)
                    else:
                        prSentence += np.log2(self.unigram[testingData[sentence][word]]) -  np.log2(unigramSize)
                        # print('word pr' , self.unigram[testingData[sentence][word]] / unigramSize)
                        # print('sentence pr', prSentence)
                prSentence += np.log2(self.unigram['<STOP>']) - np.log2(unigramSize)
                
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
            prSum = 0
            testTotalTokens=0
            for sentence in range(0,len(testingData)):
                testTotalTokens += len(testingData[sentence]) + 1
                #calculate pr of each word
                prSentence = 0
                for word in range(0,len(testingData[sentence])):
                    if word == 0:
                        if self.unigram[testingData[sentence][word]] == 0:
                            currWord = '<UNK>'
                        else:
                            currWord = testingData[sentence][word]

                        if self.bigram[('<START>', currWord)] != 0:
                            prSentence += np.log2(self.bigram[('<START>', currWord)]) - np.log2(self.unigram[self.stop_token])

                    if word == len(testingData[sentence]) - 1:
                        if self.unigram[testingData[sentence][word]] == 0:
                            prevWord = '<UNK>'
                        else:
                            prevWord = testingData[sentence][word]
                        if self.bigram[(prevWord, '<STOP>')] != 0 and self.unigram[prevWord] != 0:
                            prSentence += np.log2(self.bigram[(prevWord, '<STOP>')]) - np.log2(self.unigram[prevWord])
                    else:
                        if self.unigram[testingData[sentence][word+1]] == 0:
                            currWord = '<UNK>'
                        else:
                            currWord = testingData[sentence][word+1]
                        if self.unigram[testingData[sentence][word]] == 0:
                            prevWord = '<UNK>'
                        else:
                            prevWord = testingData[sentence][word]
                        
                        if self.bigram[(prevWord, currWord)] != 0 and self.unigram[prevWord] != 0:
                            prSentence += np.log2(self.bigram[(prevWord, currWord)]) - np.log2(self.unigram[prevWord])         

                prSum += prSentence

            # print("testTotalTokens=", testTotalTokens)
            # print("sum" , prSum)
            L = prSum / testTotalTokens
            # print("L", L)
            perplexity = np.power(2, -1 * L)
            return perplexity
        
        #TRIGRAM PERPLEXITY
        elif self.n_grams == 3:
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
                        if self.unigram[testingData[sentence][word]] == 0:
                            currWord = '<UNK>'
                        else: currWord = testingData[sentence][word]
                        prevWord1 = '<START>'
                        prevWord2 = '<PRESTART>'
                        # print('prevWord1: word at index ', word, 'is: ', testingData[sentence][word])
                        # print('currWord: word at index ', word, '+ 1 is: ', testingData[sentence][word+1])
                        # print('currWord is: ', currWord, 'prevWord1 is: ', '<START>', 'prevWord2 is: ', '<PRESTART>')
                        # print('trigram count', self.trigram[('<PRESTART>', '<START>', currWord)])
                        # print('bigram count', self.bigram[('<START>', currWord)])
                        if self.trigram[('<PRESTART>', '<START>', currWord)] != 0:
                            # bigram denominator for first start token is start start, equiv to # of stop tokens
                            prSentence += np.log2(self.trigram[('<PRESTART>', '<START>', currWord)]) - np.log2(self.unigram['<STOP>'])

                        if self.unigram[testingData[sentence][word+1]] == 0:
                            currWord = '<UNK>'
                        else: currWord = testingData[sentence][word+1]
                        if self.unigram[testingData[sentence][word]] == 0:
                            prevWord1 = '<UNK>'
                        else: prevWord1 = testingData[sentence][word]
                        prevWord2 = '<START>'
                        # print('prevWord1: word at index ', word, 'is: ', testingData[sentence][word])
                        # print('currWord: word at index ', word, '+ 1 is: ', testingData[sentence][word+1])
                        # print('currWord is: ', currWord, 'prevWord1 is: ', prevWord1, 'prevWord2 is: ', '<START>')
                        # print('trigram count', self.trigram[('<START>', prevWord1, currWord)])
                        # print('bigram count', self.bigram[('<START>', prevWord1)])
                        if self.trigram[('<START>', prevWord1, currWord)] != 0 and self.bigram[('<START>', prevWord1)] != 0:
                            prSentence += np.log2(self.trigram[('<START>', prevWord1, currWord)]) - np.log2(self.bigram[('<START>', prevWord1)])


                    elif word == 0 and len(testingData[sentence]) == 1:
                        if self.unigram[testingData[sentence][word]] == 0: currWord = '<UNK>'
                        else: currWord = testingData[sentence][word]
                        prevWord1 = '<START>'
                        prevWord2 = '<PRESTART>'
                        # print('prevWord1: word at index ', word, 'is: ', testingData[sentence][word])
                        # print('currWord: word at index ', word, '+ 1 is: ', testingData[sentence][word+1])
                        # print('currWord is: ', currWord, 'prevWord1 is: ', '<START>', 'prevWord2 is: ', '<PRESTART>')
                        # print('trigram count', self.trigram[('<PRESTART>', '<START>', currWord)])
                        # print('bigram count', self.bigram[('<START>', currWord)])
                        if self.trigram[('<PRESTART>', '<START>', currWord)] != 0 and self.bigram[('<START>', currWord)] != 0:
                            prSentence += np.log2(self.trigram[('<PRESTART>', '<START>', currWord)]) - np.log2(self.bigram[('<START>', currWord)])

                        if self.unigram[testingData[sentence][word]] == 0:
                            prevWord1 = '<UNK>'
                        else: prevWord1 = testingData[sentence][word]
                        currWord = '<STOP>'
                        prevWord2 = '<START>'
                        # print('prevWord1: word at index ', word, 'is: ', testingData[sentence][word])
                        # print('currWord: word at index ', word, '+ 1 is: ', testingData[sentence][word+1])
                        # print('currWord is: ', currWord, 'prevWord1 is: ', prevWord1, 'prevWord2 is: ', '<START>')
                        # print('trigram count', self.trigram[('<START>', prevWord1, currWord)])
                        # print('bigram count', self.bigram[('<START>', prevWord1)])
                        if self.trigram[('<START>', prevWord1, currWord)] != 0 and self.bigram[('<START>', prevWord1)] != 0:
                            prSentence += np.log2(self.trigram[('<START>', prevWord1, currWord)]) - np.log2(self.bigram[('<START>', prevWord1)])
                            
                    elif word == len(testingData[sentence]) - 1:
                        # currWord = self.stop_token
                        if self.unigram[testingData[sentence][word]] == 0:
                            prevWord1 = '<UNK>'
                        else: prevWord1 = testingData[sentence][word]
                        if self.unigram[testingData[sentence][word-1]] == 0:
                            prevWord2 = '<UNK>'
                        else: prevWord2 = testingData[sentence][word-1]
                        # print('prevWord1: word at index ', word, 'is: ', testingData[sentence][word])
                        # print('prevWord2: word at index ', word, '- 1 is: ', testingData[sentence][word-1])
                        # print('currWord is: ', '<STOP>', 'prevWord1 is: ', prevWord1, 'prevWord2 is: ', prevWord2)
                        # print('trigram count', self.trigram[(prevWord2, prevWord1, '<STOP>')])
                        # print('bigram count', self.bigram[(prevWord2, prevWord1)])
                        if self.trigram[(prevWord2, prevWord1, '<STOP>')] != 0 and self.bigram[(prevWord2, prevWord1)] != 0:
                            prSentence += np.log2(self.trigram[(prevWord2, prevWord1, '<STOP>')]) - np.log2(self.bigram[(prevWord2, prevWord1)])
                    else:
                        if self.unigram[testingData[sentence][word+1]] == 0:
                            currWord = '<UNK>'
                        else: currWord = testingData[sentence][word+1]

                        if self.unigram[testingData[sentence][word]] == 0:
                            prevWord1 = '<UNK>'
                        else:  prevWord1 = testingData[sentence][word]
                        
                        if self.unigram[testingData[sentence][word-1]] == 0:
                            prevWord2 = '<UNK>'
                        else: prevWord2 = testingData[sentence][word-1]
                        
                        # print('prevWord2: word at index ', word, '- 1 is: ', testingData[sentence][word-1])
                        # print('prevWord1: word at index ', word, 'is: ', testingData[sentence][word])
                        # print('currWord: word at index ', word, '+ 1 is: ', testingData[sentence][word+1])
                        # print('currWord is: ', currWord, 'prevWord1 is: ', prevWord1, 'prevWord2 is: ', prevWord2)
                        # print('trigram count', self.trigram[(prevWord2, prevWord1, currWord)])
                        # print('bigram count', self.bigram[(prevWord2, prevWord1)])
                        if self.trigram[(prevWord2, prevWord1, currWord)] != 0 and self.bigram[(prevWord2, prevWord1)] != 0:
                            prSentence += np.log2(self.trigram[(prevWord2, prevWord1, currWord)]) - np.log2(self.bigram[(prevWord2, prevWord1)])        

                prSum += prSentence

            # print("testTotalTokens=", testTotalTokens)
            # print("sum" , prSum)
            L = prSum / testTotalTokens
            # print("L", L)
            perplexity = np.power(2, -1 * L)
            return perplexity



