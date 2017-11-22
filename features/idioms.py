###### idiom source:
#         https://www.englishclub.com/ref/Idioms/
# simplifications:
#      (word) -> deleted
#      as soon as possible | asap -> as soon as possible
#                                    asap
#      punctuations were deleted at the end of idioms
##############################################################

from sklearn.base import BaseEstimator, TransformerMixin
from collections import defaultdict


idioms_list = list( set( idiom.strip().lower() for idiom in open('features/idioms.txt', 'r').readlines() ) )

class Idioms (BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass

    def fit(self, x, y):
        return self
        
    def transform(self, x):
        idioms = idioms_list
        features =[]
        for pun in x:
            pun = pun.strip().lower()
            temp = {'idiom':0}
            for idiom in idioms: 
                if idiom in pun:
                    temp['idiom'] = 1       
            features.append(temp)


        return features
