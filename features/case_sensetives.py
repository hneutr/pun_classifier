from sklearn.base import BaseEstimator, TransformerMixin
from collections import defaultdict


class AllUpper(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass

    def fit(self, x,y):
        return self
    def transform(self, x):
        features =[]
        for pun in x:
            pun = pun.strip().split()
            temp = defaultdict(int)
            for word in pun:
                if word[0].isupper():
                    temp[word] += 1
                elif word[0].capitalize():
                    temp[word] += 1
            features.append(temp)
        return features
