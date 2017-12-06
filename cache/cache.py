import pickle
from pathlib import Path

class Cache:

    def __init__(self):
        pass

    # Assume cache exists for classifier if there is a pkl file for it
    def has(self, classifier):
        my_file = Path(self._get_filename(classifier))
        return my_file.is_file()

    def get(self, classifier):
        return pickle.load(open(self._get_filename(classifier), "rb"))

    def set(self, classifier):
        # Some classifiers may not cache for the time being
        if not hasattr(classifier, 'no_cache'):
            f = open(self._get_filename(classifier), "wb")
            pickle.dump(classifier, f)

    def _get_filename(self, classifier):
        return './cache/%s.pkl' % classifier.name
