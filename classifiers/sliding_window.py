from nltk import ClassifierBasedTagger, MaxentClassifier
from nltk.chunk.named_entity import shape
from nltk import download
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem.snowball import SnowballStemmer
import numpy as np
download('words')
download('averaged_perceptron_tagger')



class PunSlidingWindowClassifier(ClassifierBasedTagger):

    def __init__(self, output="word"):
        self.name = "Sliding Window"
        self.output = output


    def train(self, x_train, y_train):
        # make y one-hot
        y_train = np.asarray([np.eye(1, len(x), y)[0] for x, y in zip(x_train, y_train)])

        train = [list(zip(a[0], a[1])) for a in zip(x_train, y_train)]
        ClassifierBasedTagger.__init__(
            self, train=train,
            classifier_builder=self._classifier_builder)

        # Give training predictions so we can evaluate training accuracy
        return self.test(x_train)

    def _classifier_builder(self, train):
        return MaxentClassifier.train(train,  # algorithm='megam',
                                      gaussian_prior_sigma=1,
                                      trace=2)

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
        tokens = list(zip(tokens, ))
        index += 2

        # Need to pad tokens with start/end tags for when window is at start or end of sentence
        tokens = ['[START2]', '[START1]'] + list(tokens) + ['[END1]', '[END1]']
        word = tokens[index][0]
        stemmer = SnowballStemmer('english')

        features = {
            'position': index / (len(tokens)-4),
            'lemma': stemmer.stem(word),
            'shape': shape(word),
            'wordlen': len(word),
            'prefix3': word[:3].lower(),
            'suffix3': word[-3:].lower(),
            'word': word,
            'pos': pos_tag([word])[0][1],
            'en-wordlist': (word in self._english_wordlist()),
            'stopwords': word.lower() in self._stopwords(),
            'prev1word': tokens[index-1][0],
            'prev2word': tokens[index-2][0],
            'next1word': tokens[index+1][0],
            'next2word': tokens[index+2][0]
            #'word+nextpos': '{0}+{1}'.format(word.lower(), pos_tag([tokens[index+1][0]])[0][1]),
            }

        return features

    def test(self, x_test):
        return self.get_output(x_test)

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
