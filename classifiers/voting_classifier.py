from sklearn.ensemble import VotingClassifier


class PunVotingClassifier:
    def __init__(self, type, classifiers):
        self.name = "Voting %s" % type

        estimators = []
        for classifier in classifiers:
            estimators.append(('%s' % classifier.name, classifier))

        self.model = VotingClassifier(estimators=estimators, voting='hard')
        self.no_cache = True # Can't cache voting classifier due to keras model used - makes it difficult
        # But, classifiers used in this are themselves cached so not a big deal and won't take long to run

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)
        return self.model.predict(x_train)

    def test(self, x_test):
        return self.model.predict(x_test)

    def test_with_probabilities(self, x_test):
        return self.model.predict_proba(x_test)

