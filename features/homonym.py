########## homonym sources:
#               https://en.wikipedia.org/wiki/List_of_true_homonyms
#               http://home.alphalink.com.au/~umbidas/homonym_main.htm
##########################


from sklearn.base import BaseEstimator, TransformerMixin
from collections import defaultdict

homonym_list = list( set( idiom.strip().lower() for idiom in open('features/homonym.txt', 'r').readlines() ) )

class Homonym (BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass

    def fit(self, x, y):
        return self
        
    def transform(self, x):
        homonyms = homonym_list
        features =[]
        for pun in x:
            pun = pun.strip().lower()
            temp = {'homonym':0}
            for homonym in homonyms: 
                if homonym in pun:
                    temp['homonym'] = 1       
            features.append(temp)


        return features
