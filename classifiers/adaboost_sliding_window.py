from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import AdaBoostClassifier

from classifiers.scikit_wrapper import ScikitWrapperClassifier
from classifiers.sliding_window import PunSlidingWindowClassifier


class AdaboostSlidingWindowClassifier(BaseEstimator, ClassifierMixin):
    """An example of classifier"""

    def __init__(self, classifier):
        """
        Called when initializing the classifier
        """
        self.name = "Adaboost for Sliding Window"
        self.classifier = AdaBoostClassifier(base_estimator=ScikitWrapperClassifier(PunSlidingWindowClassifier(output="word")))

    def fit(self, x_train, y_train=None):

        self.classifier.train(x_train, y_train)

        return self


    def predict(self, x):
        return self.classifier.test(x)

    def score(self, X, y=None):
        #TODO
        pass