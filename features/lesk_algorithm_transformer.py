from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from nltk import download as nltk_download
nltk_download('wordnet')

class LeskAlgorithmTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, examples):
        return self

    def transform(self, sentences):
        for sentence in sentences:
            for word in sentence:
                word_syn = lesk(sentence, word)
                # TODO: What should we do with this?


        # Return some other stuff so not broken while figuring out this transformer
        features = np.zeros((len(sentences), 1))
        i = 0
        for ex in sentences:
            features[i, 0] = len(ex)
            i += 1

        return features