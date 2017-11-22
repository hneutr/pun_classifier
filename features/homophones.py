##### Sources for homophones:
#          http://www.zyvra.org/lafarr/hom.htm
#          http://www.singularis.ltd.uk/bifroest/misc/homophones-list.html
#          https://www.teachingtreasures.com.au/teaching-tools/Basic-worksheets/worksheets-english/upper/homophones-list.htm
###############################
from sklearn.base import BaseEstimator, TransformerMixin
from collections import defaultdict

homophone_words_list = list(set(line.strip().lower() for line in open('features/homophones.txt', 'r').readlines()))


class Homophone (BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass

    def fit(self, x, y):
        return self
    def transform(self, x):
        
        homophone_words = homophone_words_list
        features =[]
        for pun in x:
            pun = pun.strip().split()
            temp = defaultdict(int)
            for words in pun:
                for word in homophone_words:
                    if words.lower() == word:
                        temp[word] += 1
            features.append(temp)

        return features


class Homophone_Number (BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass

    def fit(self, x, y):
        return self
    def transform(self, x):
        
        homophone_words = homophone_words_list
        features =[]
        for pun in x:
            pun = pun.strip().split()
            temp = {'homophone':0}
            for words in pun:
                for word in homophone_words:
                    if words.lower() == word:
                        temp['homophone'] += 1
            features.append(temp)
            
        return features


class Homophone_0_1 (BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass

    def fit(self, x, y):
        return self
    def transform(self, x):
        
        homophone_words = homophone_words_list
        features =[]
        for pun in x:
            pun = pun.strip().split()
            temp = {'homophone':0}
            #temp = {}
            for words in pun:
                for word in homophone_words:
                    if words.lower() == word:
                        temp['homophone'] = 1
            features.append(temp)
            
        return features
