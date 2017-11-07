import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.metrics import accuracy_score



'''
The ItemSelector class was created by Matt Terry to help with using
Feature Unions on Heterogeneous Data Sources

All credit goes to Matt Terry for the ItemSelector class below

For more information:
http://scikit-learn.org/stable/auto_examples/hetero_feature_union.html
'''
class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


"""
This is an example of a custom feature transformer. The constructor is used
to store the state (e.g like if you need to store certain words/vocab), the
fit method is used to update the state based on the training data, and the
transform method is used to transform the data into the new feature(s). In
this example, we simply use the length of the movie review as a feature. This
requires no state, so the constructor and fit method do nothing.
"""
class TextLengthTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, examples):
        return self

    def transform(self, examples):
        features = np.zeros((len(examples), 1))
        i = 0
        for ex in examples:
            features[i, 0] = len(ex)
            i += 1

        return features



class Featurizer:
    def __init__(self):
        # To add new features, just add a new pipeline to the feature union
        # The ItemSelector is used to select certain pieces of the input data
        # In this case, we are selecting the plaintext of the input data
        self.all_features = FeatureUnion(
            transformer_list= [
                ('text_stats', Pipeline([
                    ('selector', ItemSelector(key='tokens')),
                    ('token_count', TextLengthTransformer())
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
        self.model = SGDClassifier(loss='log', penalty='l2', alpha=0.0001, max_iter=15000, shuffle=True, verbose=2)

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

    def test(self, x_test, y_test):
        # Here we collect the test features
        feat_test = self.feat.test_feature({
            'tokens': x_test
        })
        y_pred = self.model.predict(feat_test)
        accuracy = accuracy_score(y_pred, y_test)
        print("Accuracy on test set =", accuracy)

