from collections import Counter, defaultdict


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

                # if we reached the last word
                if word == len(data[sentence]) - 1:
                    word1 = data[sentence][word]
                    word2 = self.stop_token
                    self.bigram[(word1, word2)] += 1
                else:
                    word1 = data[sentence][word]
                    word2 = data[sentence][word+1]
                    self.bigram[(word1, word2)] += 1
        # print(self.bigram)

    def unigramExtract(self, data):
        for sentence in range(0, len(data)):
            for word in range(0, len(data[sentence])):
                self.unigram[data[sentence][word].lower()] += 1
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
        print(data)
        # print("done and n_grams is:", self.n_grams)

        #TRACK COUNTS/PROBABILITIES 
        #use counter dictionary, ngram key, count value

        # UNIGRAM EXTRACTION
        if (self.n_grams == 1):
            self.unigramExtract(data)
            print("unigram counter length = ", len(self.unigram))
            print(self.unigram)

        # BIGRAM EXTRACTION
        elif (self.n_grams == 2):
            self.unigramExtract(data)
            self.bigramExtract(data)

        # TRIGRAM EXTRACTION
        elif (self.n_grams == 3):
            self.unigramExtract(data)
            self.bigramExtract(data)
            self.trigramExtract(data)
            
        return 
