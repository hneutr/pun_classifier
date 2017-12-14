from keras.models import Sequential
from keras.layers import Dense, Embedding, Bidirectional, Dropout
from keras.layers import LSTM, Input, TimeDistributed
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from word_embeddings import WordEmbeddings
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

MAX_NB_WORDS = 20000

class PunRNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, output="word"):
        """
        output can be one of:
            - word
            - sequence
            - binary
        """

        self.name = "Pun RNN Location"
        self.embedding = WordEmbeddings()
        self.output = output
        self.use_keras_caching = True

    def train(self, x_train, y_train):
        # make y_train into a 1-hot vector
        self.y_train = np.asarray([np.eye(1, len(x), y)[0] for x, y in zip(x_train, y_train)])

        self.fit_xs(x_train)
        self.x_train = self.format_xs(x_train)

        num_words = min(MAX_NB_WORDS, len(self.word_index) + 1)
        embedding_matrix = self.embedding.get_matrix(self.word_index, num_words)

        self.model = Sequential()
        self.model.add(
            Embedding(
                num_words,
                self.embedding.embedding_length,
                weights = [ embedding_matrix ],
            )
        )
        self.model.add(Dropout(.2))
        self.model.add(Bidirectional(
            LSTM(128, dropout=.8, input_dim=300, return_sequences=True),
            merge_mode='ave'
        ))
        self.model.add(TimeDistributed(Dense(1, activation='sigmoid')))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        for i in range(1, 7):
            print("epoch ", i)
            for x, y in zip(self.x_train, self.y_train):
                x = x.reshape(1, len(x))
                y = y.reshape(1, len(y), 1)

                self.model.fit(x, y, batch_size=1, epochs=1, verbose=0)

        return self.get_output(self.x_train)


    def get_output(self, xs):
        if self.output == "word":
            return self.get_word_predictions(xs)
        elif self.output == "sequence":
            return self.get_sequence_predictions(xs)
        elif self.output == "binary":
            return self.get_binary_predictions(xs)

    def get_binary_predictions(self, xs):
        predictions = []
        for x in xs:
            x = x.reshape(1, len(x))
            prediction = self.model.predict(x, batch_size=1, verbose=0)[0]

            # magic, I don't know (why this needs to be done, that is)
            prediction = [p[0] for p in prediction]

            is_pun = 1 if len([ 1 for x in prediction if x > .5 ]) else 0
            predictions.append(is_pun)

        return predictions

    def get_predictions_with_probabilities(self, xs):
        predictions = []
        for x in xs:
            x = x.reshape(1, len(x))
            prediction = self.model.predict(x, batch_size=1, verbose=0)[0]

            # magic, I don't know (why this needs to be done, that is)
            prediction = [p[0] for p in prediction]

            predictions.append(prediction)

        return predictions

    def get_word_predictions(self, xs):
        predictions = []
        for x in xs:
            x = x.reshape(1, len(x))
            prediction = self.model.predict(x, batch_size=1, verbose=0)[0]

            # magic, I don't know (why this needs to be done, that is)
            prediction = [p[0] for p in prediction]

            word_index = np.argmax(prediction)
            predictions.append(word_index)

        return predictions

    def get_sequence_predictions(self, xs):
        predictions = []
        for x in xs:
            x = x.reshape(1, len(x))
            prediction = self.model.predict_classes(x, batch_size=1, verbose=0)[0]

            # magic, I don't know (why this needs to be done, that is)
            prediction = [p[0] for p in prediction]
            predictions.append(prediction)

        return predictions


    def test(self, x_test):
        self.x_test = self.format_xs(x_test)

        return self.get_output(self.x_test)

    def test_with_probabilities(self, x_test):
        self.x_test = self.format_xs(x_test)

        return self.get_predictions_with_probabilities(self.x_test)

    def format_xs(self, xs):
        return np.asarray([np.asarray([self.word_index.get(t.lower(), 0) for t in x]) for x in xs])

    def fit_xs(self, xs):
        tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
        tokenizer.fit_on_texts([' '.join(x) for x in xs])
        self.word_index = tokenizer.word_index

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
