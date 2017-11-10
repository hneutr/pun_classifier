import argparse
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

SEED = 20171109

class Data:
    def __init__(self, path):
        with open(path, 'rb') as f:
            x_set, y_set = pickle.load(f)

            max_len = max([len(x) for x in x_set])

            train_x, valid_x, self.train_y, self.valid_y = train_test_split(x_set, y_set, random_state=SEED)

            self.train_x = [ " ".join(x) for x in train_x ]
            self.valid_x = [ " ".join(x) for x in valid_x ]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='baseline multinomial naive bayes')
    parser.add_argument('--graphic', type=str, default='homographic', help="which type of pun ['homographic', 'heterographic']. Default: homographic")

    args = parser.parse_args()

    data = Data("data/pickles/test-1-%s.pkl.gz" % args.graphic)

    nb = MultinomialNB()
    
    count_vec = CountVectorizer()
    x_train_counts = count_vec.fit_transform(data.train_x)
    x_valid_counts = count_vec.transform(data.valid_x)

    nb.fit(x_train_counts, data.train_y)
    score = nb.score(x_valid_counts, data.valid_y)
    print(score)
