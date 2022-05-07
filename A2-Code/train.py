from collections import Counter, defaultdict


class NGramLM(object):
    def __init__(self, n_grams=1):
        self.n_grams = n_grams
        self.unigram = Counter()

    def train(self, file_path, needs_preprocess=True):

        #TOKENIZING
        with open(file_path, 'r') as file:
            data = file.readlines()
            for sentence in range(0, len(data)):
                # print(data[sentence])
                data[sentence] = data[sentence].strip(' \n').split(' ') #BUG: empty new lines are being stored
        
        print(data)

        #PREPROCESSING (OPTIONAL)
        #need to unkify words (OOV if less than 3 counts)
        if(needs_preprocess):
            with open("./data/processed.1b_benchmark.train.tokens", 'w') as file:
                for sentence in range(0, len(data)):
                    for word in range(0, len(data[sentence])):
                        file.write(data[sentence][word])
                        if word != len(data[sentence]) - 1:
                            file.write(' ')
                        else:
                            file.write('\n')
        
        # print("done and n_grams is:", self.n_grams)

        #TRACK COUNTS/PROBABILITIES 
        #use counter dictionary, ngram key, count value
        # how to adjust for ngrams? since key goes from 'Xi' to 'Xi-1 Xi' to 'Xi-2 Xi-1 Xi'
        for sentence in range(0, len(data)):
            for word in range(0, len(data[sentence])):
                self.unigram[data[sentence][word].lower()] += 1
                if word != len(data[sentence]) - 1:
                    self.unigram['<STOP>'] += 1

        print(self.unigram)

        return 
