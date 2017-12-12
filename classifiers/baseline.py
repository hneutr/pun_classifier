from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, ClassifierMixin

class BaselinePunClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, type):
        self.name = "Baseline %s" % type
        self.model = MultinomialNB()
        self.featurizer = CountVectorizer()
    
    def train(self, x_train, y_train):
        self.x_train = self.featurizer.fit_transform([ " ".join(x) for x in x_train ])
        self.y_train = y_train

        self.model.fit(self.x_train, self.y_train)
        return self.model.predict(self.x_train)

    def test(self, x_test):
        self.x_test = self.featurizer.transform([ " ".join(x) for x in x_test ])

        return self.model.predict(self.x_test)

    def test_with_probabilities(self, x_test):
        self.x_test = self.featurizer.transform([" ".join(x) for x in x_test])
        return self.model.predict_proba(self.x_test)

    def fit(self, x_train, y_train=None):
        self.train(x_train, y_train)

        return self

    def predict(self, x):
        return self.test(x)

    def predict_proba(self, x):
        return self.test_with_probabilities(x)

    def score(self, x, y, sample_weight=None):
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(x), sample_weight=sample_weight)


