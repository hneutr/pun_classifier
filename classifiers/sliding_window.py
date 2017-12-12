from nltk import ClassifierBasedTagger, MaxentClassifier
from nltk.chunk.named_entity import shape
from nltk import download
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem.snowball import SnowballStemmer
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class PunSlidingWindowClassifier(ClassifierBasedTagger, BaseEstimator, ClassifierMixin):
    def __init__(self, output="word", window=5):
        self.name = "Sliding Window"
        self.output = output
        self.stemmer = SnowballStemmer('english')
        self.window = window

    def train(self, x_train, y_train):
        y_train = np.asarray([np.eye(1, len(x), y)[0] for x, y in zip(x_train, y_train)])

        x_train = self.format_xs(x_train)

        train = [list(zip(a[0], a[1])) for a in zip(x_train, y_train)]
        self.classifier = ClassifierBasedTagger.__init__(
            self, train=train,
            classifier_builder=self._classifier_builder)

        # Give training predictions so we can evaluate training accuracy
        return self.get_output(x_train)

    def _classifier_builder(self, train):
        return MaxentClassifier.train(train,  # algorithm='megam',
                                      gaussian_prior_sigma=1,
                                      trace=2)

    def format_xs(self, xs):
        return [ pos_tag(x) for x in xs ]

    def _english_wordlist(self):
        try:
            wl = self._en_wordlist
        except AttributeError:
            from nltk.corpus import words
            self._en_wordlist = set(words.words('en-basic'))
            wl = self._en_wordlist
        return wl

    def _stopwords(self):
        try:
            sl = self._stopword_list
        except AttributeError:
            self._stopword_list = set(stopwords.words('english'))
            sl = self._stopword_list
        return sl

    def _feature_detector(self, tokens, index, history):
        index += self.window

        right_padding = [('[START%d]' % i, '<S{%d}>' % i) for i in range(self.window, 0, -1)]
        left_padding = [('[END{%d}]' % i, '<E{%d}>' % i) for i in range(self.window, 0, -1)]

        # Need to pad tokens with start/end tags for when window is at start or end of sentence
        tokens = right_padding + list(tokens) + left_padding
        word = tokens[index][0]

        features = {
            'position': index / len(tokens),
            'words_remaining': (len(tokens) - 2 * self.window) - (index - self.window),
            'lemma': self.stemmer.stem(word),
            'shape': shape(word),
            'wordlen': len(word),
            'prefix3': word[:3].lower(),
            'suffix3': word[-3:].lower(),
            'word': word,
            'pos': tokens[index][1],
            'en-wordlist': (word in self._english_wordlist()),
            'stopwords': word.lower() in self._stopwords(),
            }

        for i in range(1, self.window+1):
            features['prev{%d}word' % i] = tokens[index-i][0]
            features['prev{%d}pos' % i] = tokens[index-i][1]
            features['next{%d}word' % i] = tokens[index+i][0]
            features['next{%d}pos' % i] = tokens[index+i][1]

        return features

    def test(self, xs):
        xs = self.format_xs(xs)
        return self.get_output(xs)

    def test_with_probabilities(self, xs):
        xs = self.format_xs(xs)
        return [self.prob_tag(sent) for sent in xs]

    def prob_tag(self, tokens):
        # docs inherited from TaggerI
        tags = []
        for i in range(len(tokens)):
            tags.append(self.choose_tag_prob(tokens, i, tags))
        return tags

    def choose_tag_prob(self, tokens, index, history):
        # Use our feature detector to get the featureset.
        featureset = self._feature_detector(tokens, index, history)

        pdist = self._classifier.prob_classify(featureset)
        return pdist.prob(1)



    def get_output(self, xs):
        tagged_sents = [[t[1] for t in sent] for sent in self.tag_sents(xs)]

        if self.output == "word":
            return self.get_word_predictions(tagged_sents)
        elif self.output == "sequence":
            return tagged_sents
        elif self.output == "binary":
            return self.get_binary_predictions(tagged_sents)

    def get_binary_predictions(self, tagged_sents):
        predictions = []

        for sent in tagged_sents:
            is_pun = 1 if len([1 for t in sent if t > .5]) else 0

            predictions.append(is_pun)

        return predictions


    def get_word_predictions(self, tagged_sents):
        return [np.argmax(sent) for sent in tagged_sents]

    def fit(self, x_train, y_train=None):

        self.train(x_train, y_train)

        return self

    def predict(self, x):
        return self.test(x)

    def predict_proba(self, x):
        return self.test_with_probabilities(x)

    def score(self, x, y, sample_weight=None):
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(x), sample_weight=sample_weight)

