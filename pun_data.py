# Load /split training/testing data

import pickle
from sklearn.model_selection import train_test_split

SEED = 20171110

class HeterographicData:

    def __init__(self):
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []

        with open("./data/pickles/trial-1-heterographic.pkl.gz", 'rb') as f:
            self.x_train, self.y_train = pickle.load(f)

        with open("./data/pickles/test-1-heterographic.pkl.gz", 'rb') as f:
            self.x_test, self.y_test = pickle.load(f)


        # Split dataset
        #  X_train, X_test, y_train, y_test = train_test_split(dataset_x, dataset_y, test_size=0.3, random_state=SEED)


class HomographicData:

    def __init__(self):
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []

        with open("./data/pickles/test-1-homographic.pkl.gz", 'rb') as f:
            x_set, y_set = pickle.load(f)
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x_set, y_set, random_state=SEED)
