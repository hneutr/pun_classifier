import numpy as np
from functools import reduce

class WordEmbeddings:

    def __init__(self, embedding_path="data/glove.6B.300d.txt"):
        self.embeddings = {}
        self.embedding_length = 300

        with open(embedding_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            separated = line.split()

            self.embeddings[ separated[0] ] = np.asarray(separated[1:], dtype='float32')

    def embed(self, tokens, max_len=100):
        """
        given a list of tokens and a max_len, returns a 1D array of length
        embedding_len * max_len
        """
        self.max_len = max_len

        embedded = []
        for i in range(self.max_len):
            token = tokens[i] if i < len(tokens) else None
            embedded.append(self.embeddings.get(token, np.zeros((self.embedding_length))))

        return reduce(lambda x, y: x + y, embedded)
