import nltk
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from word_embeddings import WordEmbeddings
from features.homophones import *
from features.idioms import *
from features.antonym import *
from features.homonym import *

Antonyms = make_raw_antonym_list()

class PunLocationWithFeaturesClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, output = "word"):

        self.name = "Pun Location with Features"
        self.embedding = WordEmbeddings()
        self.output = output
        self.no_cache = True

    def train(self, x_train, y_train):

        #y_train = np.asarray([np.eye(1, len(x), y)[0] for x, y in zip(x_train, y_train)])
        y_predicted = []
        flag = 0
        score = 1000
        prevscore = 0
        for i in range(0, len(x_train)):
            xhalf = x_train[i][int(len(x_train[i])/2):len(x_train[i])]
            xhalf = xhalf[::-1]
            for j in range(0, len(xhalf)):
                if xhalf[j] in homophone_words_list:
                    argpun = int(len(x_train[i])/2) + len(xhalf) - 1 - j
                    flag = 1
                    break
                if xhalf[j] in homonym_list:
                    argpun = int(len(x_train[i])/2) + len(xhalf) - 1 - j
                    flag = 1
                    break
                if xhalf[j] in Antonyms:
                    argpun = int(len(x_train[i])/2) + len(xhalf) - 1 - j
                    flag = 1
                    break
                if xhalf[j] in idioms_list:
                    argpun = int(len(x_train[i])/2) + len(xhalf) - 1 - j
                    flag = 1
                    break
            if flag == 0:
                prevscore = 0
                xrev = x_train[i][::-1]
                wpair = list(nltk.bigrams(xrev))
                for i in range(0, len(wpair)):
                    score = nltk.edit_distance(wpair[i][0],wpair[i][1])
                    if score < prevscore:
                        prevscore = score
                        argpun = len(xrev) - 1 - j
            if prevscore == 0:
                argpun = len(x_train[i]) - 2
            y_predicted.append(argpun)

        return y_predicted

    def test(self, x_test):
        return self.train(x_test, None)

    def fit(self, x_train, y_train=None):

        self.train(x_train, y_train)

        return self

    def predict(self, x):
        return self.test(x)

    def score(self, x, y, sample_weight=None):
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(x), sample_weight=sample_weight)
