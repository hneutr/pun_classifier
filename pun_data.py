# Load /split training/testing data

import pickle
from sklearn.model_selection import train_test_split
import numpy as np

SEED = 20171110

class DetectionData:
    def __init__(self, graphic, even=False):
        path = "./data/pickles/test-1-%s" % graphic

        if even:
            path += "-even"
        
        path += ".pkl.gz"

        with open(path, 'rb') as f:
            self.x_set, self.y_set = pickle.load(f)
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_set, self.y_set, random_state=SEED)


class LocationData:
    def __init__(self, graphic):
        path = "./data/pickles/test-2-%s" % graphic

        path += ".pkl.gz"

        with open(path, 'rb') as f:
            self.x_set, self.y_set = pickle.load(f)

            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_set, self.y_set,
                                                                                    random_state=SEED)
