from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.metrics import accuracy_score

from features.item_selector import ItemSelector
from features.lesk_algorithm_transformer import LeskAlgorithmTransformer

class Featurizer:
    def __init__(self):
        # To add new features, just add a new pipeline to the feature union
        # The ItemSelector is used to select certain pieces of the input data
        # In this case, we are selecting the plaintext of the input data
        self.all_features = FeatureUnion(
            transformer_list= [
                ('lesk_algorithm', Pipeline([
                    ('selector', ItemSelector(key='tokens')),
                    ('lesk', LeskAlgorithmTransformer())
                ]))
            ]
        )

    def train_feature(self, examples):
        return self.all_features.fit_transform(examples)

    def test_feature(self, examples):
        return self.all_features.transform(examples)


# Pun detection classifier using feature engineering
class PunDetectionWithFeaturesClassifier:
    def __init__(self):
        self.feat = Featurizer()

        # TODO: Which classifier should be used here? Also probably want to do cross-validation on hyperparameters.
        self.model = SGDClassifier(loss='log', penalty='l2', alpha=0.0001, max_iter=15000, shuffle=True, verbose=2)

    # Train the classifier using x_train which is the set of sentences
    # And y_train which is the set of labels for those sentences
    def train(self, x_train, y_train):
        # Here we collect the train features
        # The inner dictionary contains certain pieces of the input data that we
        # would like to be able to select with the ItemSelector
        # The text key refers to the plaintext
        feat_train = self.feat.train_feature({
            'tokens': x_train
        })

        self.model.fit(feat_train, y_train)
        y_pred = self.model.predict(feat_train)
        accuracy = accuracy_score(y_pred, y_train)
        print("Accuracy on training set =", accuracy)

    # Make predictions on test data. Use the y_test labels to find accuracy of predictions.
    def test(self, x_test, y_test):
        # Here we collect the test features
        feat_test = self.feat.test_feature({
            'tokens': x_test
        })
        y_pred = self.model.predict(feat_test)
        accuracy = accuracy_score(y_pred, y_test)
        print("Accuracy on test set =", accuracy)

