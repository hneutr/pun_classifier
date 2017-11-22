from sklearn.base import BaseEstimator, TransformerMixin
from collections import defaultdict

class Unigram(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass

    def fit(self, x, y):
        return self
    def transform(self, x):
        features =[]
        for pun in x:
            pun = pun.strip().split()
            temp = defaultdict(int)
            for word in range(0, len(pun)):
                temp[(pun[word])] +=1
            features.append(temp) 
        return features


class Bigrams(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass

    def fit(self, x, y):
        #print (type(x))
        return self
    def transform(self, x):
        features =[]
        for pun in x:
            pun = pun.strip().split()
            temp = {}
            for word in range(0, len(pun)-1):
                if (pun[word], pun[word+1]) in temp:
                    temp[(pun[word], pun[word+1])] +=1
                else: 
                    temp[(pun[word], pun[word+1])] =1
            features.append(temp) 
        return features


class Trigrams(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass

    def fit(self, x, y):
        return self
    def transform(self, x):
        features =[]
        for pun in x:
            pun = pun.strip().split()
            temp = {}
            for word in range(0, len(pun)-2):
                if (pun[word], pun[word+1],pun[word+2]) in temp:
                    temp[(pun[word], pun[word+1],pun[word+2])] +=1
                else: 
                    temp[(pun[word], pun[word+1],pun[word+2])] =1
            features.append(temp) 
        return features
