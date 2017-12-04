from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from nltk.wsd import lesk
from nltk import download as nltk_download

class LeskAlgorithmTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, max_length=100):
        self.max_length = max_length
        pass

    def fit(self, xs, ys):
        return self

    def transform(self, sentences):
        features = []
        for sentence in sentences:
            # Get the synset for each word in the sentence
            word_syns = list(map(lambda word: lesk(sentence, word), sentence))

            # Create a flat matrix of how similar each word in the sentence is to every other word in the sentence
            # http://www.nltk.org/howto/wsd.html
            similarities = np.zeros(self.max_length)
            for index_a, word_syn_a in enumerate(word_syns):
                for index_b, word_syn_b in enumerate(word_syns):
                    if index_a is not index_b and len(similarities) < self.max_length:
                        if word_syn_a is None or word_syn_b is None:
                            similarities = np.append(similarities, 0)
                        else:
                            # Use the path_similarity to compute the similarities between two synsets
                            similarity = word_syn_a.path_similarity(word_syn_b)
                            similarities = np.append(similarities, similarity if similarity is not None else 0)

            features.append(similarities)


        return features
