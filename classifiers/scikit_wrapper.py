from sklearn.base import BaseEstimator, ClassifierMixin



class ScikitWrapperClassifier(BaseEstimator, ClassifierMixin):
    """An example of classifier"""

    def __init__(self, classifier):
        """
        Called when initializing the classifier
        """
        self.name = "Scikit Wrapper for " + classifier.name
        self.classifier = classifier

    def fit(self, x_train, y_train=None):

        self.classifier.train(x_train, y_train)

        return self


    def predict(self, x):
        return self.classifier.test(x)

    def score(self, X, y=None):
        #TODO
        pass