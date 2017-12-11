from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import AdaBoostClassifier

from classifiers.scikit_wrapper import ScikitWrapperClassifier
from classifiers.sliding_window import PunSlidingWindowClassifier

# Warning! Not done - still a work in progress
class AdaboostSlidingWindowClassifier(BaseEstimator, ClassifierMixin):
    """An example of classifier"""

    def __init__(self):
        """
        Called when initializing the classifier
        """
        self.name = "Adaboost for Sliding Window"
        self.classifier = AdaBoostClassifier(base_estimator=ScikitWrapperClassifier(PunSlidingWindowClassifier(output="word")))

    def train(self, x_train, y_train):
        self.classifier.fit(x_train, y_train)
        return self.classifier.predict(x_train)

    def test(self, x_test):
        return self.classifier.predict(x_test)

    def test_with_probabilities(self, x_test):
        return self.classifier.predict_proba(x_test)