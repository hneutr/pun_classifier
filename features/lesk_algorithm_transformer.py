from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from nltk.wsd import lesk
from nltk import download as nltk_download
nltk_download('wordnet')

class LeskAlgorithmTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, examples):
        return self

    def transform(self, sentences):
        features = []
        for sentence in sentences:
            # Get the synset for each word in the sentence
            word_syns = list(map(lambda word: lesk(sentence, word), sentence))

            # Create a matrix of how similar each word in the sentence is to every other word in the sentence
            # http://www.nltk.org/howto/wsd.html
            word_similarities = []
            for word_syn_a in word_syns:
                similarities = []
                for word_syn_b in word_syns:
                    if word_syn_a is None or word_syn_b is None:
                        similarities.append(0)
                    else:
                        # Use the path_similarity to compute the similarities between two synsets
                        similarities.append(word_syn_a.path_similarity(word_syn_b))
                word_similarities.append(similarities)

            features.append(word_similarities)
            # But now we have a 3d matrix not a 2d matrix that needs to be returned by the transformer
            # What should we do?


        # Return some other stuff so not broken while figuring out this transformer
        # Code below should be removed
        features = np.zeros((len(sentences), 1))
        i = 0
        for ex in sentences:
            features[i, 0] = len(ex)
            i += 1

        return features