import pickle
from pathlib import Path
from keras.models import load_model
import copy

class Cache:

    def __init__(self):
        pass

    # Assume cache exists for classifier if there is a pkl file for it
    def has(self, classifier):
        if hasattr(classifier, 'use_keras_caching'):
            my_model_file = Path(self._get_filename(classifier) + " Model")
            my_classifier_file = Path(self._get_filename(classifier) + " Classifier")
            return my_model_file.is_file() and my_classifier_file.is_file()
        else:
            my_file = Path(self._get_filename(classifier))
            return my_file.is_file()

    def get(self, classifier):
        if hasattr(classifier, 'use_keras_caching'):
            classifier = pickle.load(open(self._get_filename(classifier) + " Classifier", "rb"))
            classifier.model = load_model(self._get_filename(classifier) + " Model")
            return classifier

        else:
            return pickle.load(open(self._get_filename(classifier), "rb"))

    def set(self, classifier):
        if hasattr(classifier, 'use_keras_caching'):
            model = classifier.model
            classifier.model = None
            f = open(self._get_filename(classifier) + " Classifier", "wb")
            pickle.dump(classifier, f)

            model.save(self._get_filename(classifier) + " Model")
            classifier.model = model

        # Some classifiers may not cache for the time being
        elif not hasattr(classifier, 'no_cache'):
            f = open(self._get_filename(classifier), "wb")
            pickle.dump(classifier, f)

    def _get_filename(self, classifier):
        return './cache/files/%s.pkl' % classifier.name
