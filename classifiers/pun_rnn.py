'''Trains an LSTM model on the IMDB sentiment classification task.
The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.
# Notes
- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.
- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
'''
from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from word_embeddings import WordEmbeddings


MAX_NB_WORDS = 20000
batch_size = 32
MAX_SEQUENCE_LENGTH = 1000
EMBEDDING_DIM = 300

class PunRNNClassifier:
    def __init__(self):
        self.name = "Pun RNN"
        self.word_embeddings = WordEmbeddings()
        self.model = Sequential()


    def train(self, x_train, y_train):
        data, word_index = self.format_data(x_train)

        # prepare embedding matrix
        num_words = min(MAX_NB_WORDS, len(word_index) + 1)
        embedding_matrix = self.word_embeddings.get_embedding_matrix(word_index, num_words, MAX_NB_WORDS)


        self.model.add(Embedding(num_words,
                                    EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    input_length=MAX_SEQUENCE_LENGTH))

        self.model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
        self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

        self.model.fit(data, y_train, batch_size=batch_size,epochs=3)
        return self.model.predict_classes(data)

    def test(self, x_test):
        data, word_index = self.format_data(x_test)
        return list(map(lambda x: x[0], self.model.predict_classes(data)))

    def format_data(self, x_data):
        tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
        sentences = list(map(lambda x: ' '.join(x), x_data))
        tokenizer.fit_on_texts(sentences)
        sequences = tokenizer.texts_to_sequences(sentences)
        data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
        return data, tokenizer.word_index
