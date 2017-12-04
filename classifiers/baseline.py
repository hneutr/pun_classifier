from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

class BaselinePunClassifier:
    def __init__(self):
        self.name = "Baseline"
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

