# Load /split training/testing data

import pickle
from sklearn.model_selection import train_test_split

SEED = 20171110

class DetectionData:
    def __init__(self, graphic, even=False, train_size=None):
        self.x_train, self.x_test, self.y_train, self.y_test = get_data(graphic, even, 1, train_size)


class LocationData:
    def __init__(self, graphic, train_size=None):
        self.x_train, self.x_test, self.y_train, self.y_test = get_data(graphic, False, 2, train_size)

def get_data (graphic, even, type, train_size):
    x_set = []
    y_set = []

    if graphic == 'combined' or graphic == 'both':
        graphics = ['heterographic', 'homographic']
        even = False # don't use even when running with both types as I think there would be some repeat data using even?
    else:
        graphics = [graphic]

    for g in graphics:

        path = "./data/pickles/test-%d-%s" % (type, g)


        if even:
            path += "-even"

        path += ".pkl.gz"

        with open(path, 'rb') as f:
            x, y = pickle.load(f)
            x_set += x

            if graphic == 'both' and g == 'heterographic':
                if type == 1:
                    y_set += list(map(lambda x: 2 if x == 1 else 0, y))
                else :
                    # Can't currently use both for location data so just use combined for now
                    y_set += y
            else:
                y_set += y

    return train_test_split(x_set, y_set, random_state=SEED, train_size=train_size)


