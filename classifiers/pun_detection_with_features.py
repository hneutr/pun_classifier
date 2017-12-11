from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.base import BaseEstimator, ClassifierMixin


from features.item_selector import ItemSelector
from features.lesk_algorithm_transformer import LeskAlgorithmTransformer
from features.raw_transformer import RawTransformer
from features.pos_transformer import PosTransformer
from word_embeddings import WordEmbeddings

from features.n_grams import Unigram, Bigrams, Trigrams
from features.negative_positive import Negatives, Positives
from features.case_sensetives import AllUpper
from features.homophones import Homophone, Homophone_Number, Homophone_0_1
from features.idioms import Idioms
from features.antonym import Antonyms
from features.homonym import Homonym



# Pun detection classifier using feature engineering
class PunDetectionWithFeaturesClassifier(BaseEstimator, ClassifierMixin):
    def tokensFunction(self, x):
        return x

    def stringFunction(self, x):
        return " ".join(x)

    def embeddingsFunction(self, x):
        return self.raw_embeddings.embed(x)

    def __init__(self, alpha=0.0001):
        self.name = "Pun Detection With Features"
        self.raw_embeddings = WordEmbeddings()
        self.alpha = alpha

        self.pipeline = Pipeline([
            (
                "raw",
                RawTransformer({
                    'tokens': self.tokensFunction,
                    'string': self.stringFunction,
                    'embeddings': self.embeddingsFunction
                })
            ),
            (
                "feature_union",
                FeatureUnion(
                    transformer_list=[
                        ('lesk_algorithm', Pipeline([
                            ('selector', ItemSelector(key='tokens')),
                            ('lesk', LeskAlgorithmTransformer(max_length=100)),
                            ('best', TruncatedSVD(n_components=50))
                        ])),
                        ('pos', Pipeline([
                            ('selector', ItemSelector(key='tokens')),
                            ('pos', PosTransformer()),
                            ('pos_tfidf', TfidfVectorizer())
                        ])),
                        ('tfidf', Pipeline([
                            ('selector', ItemSelector(key='string')),
                            ('tfidf', TfidfVectorizer()),
                            ('best', TruncatedSVD(n_components=50)),
                        ])),
                        ('embeddings', Pipeline([
                            ('selector', ItemSelector(key='embeddings')),
                        ]))
                        ,
                        #('unigram', Pipeline([
                        #    ('selector', ItemSelector(key='string')),
                        #    ('unigram for pun', Unigram()),
                        #    ('vect', DictVectorizer())
                        #]))
                        #,
                        #('bigram', Pipeline([
                        #    ('selector', ItemSelector(key='string')),
                        #    ('bigrams for pun', Bigrams()),
                        #    ('vect', DictVectorizer())
                        #])), 
                        #    ('trigram', Pipeline([
                        #    ('selector', ItemSelector(key='string')),
                        #   ('trigrams for review', Trigrams()),
                        #    ('vect', DictVectorizer())
                        #]))
                        #,
                        #('negatives', Pipeline([
                        #    ('selector', ItemSelector(key='string')),
                        #    ('trigrams for review', Negatives()),
                        #    ('vect', DictVectorizer())
                        #]))
                        #,
                        #('positives', Pipeline([
                        #    ('selector', ItemSelector(key='string')),
                        #    ('trigrams for review', Positives()),
                        #    ('vect', DictVectorizer())
                        #]))
                        #,
                        #('all_first_caps', Pipeline([
                        #    ('selector', ItemSelector(key='string')),
                        #    ('trigrams for pun', AllUpper()),
                        #('vect', DictVectorizer())
                        #])),
                        ('Homophone', Pipeline([
                            ('selector', ItemSelector(key='string')),
                            ('homophones for pun', Homophone()),
                        ('vect', DictVectorizer())
                        ]))
                        ,
                        ('Homophone_number', Pipeline([
                            ('selector', ItemSelector(key='string')),
                            ('homophones for pun', Homophone_Number()),
                        ('vect', DictVectorizer())
                        ]))
                        ,
                        ('Homophone 0 1', Pipeline([
                            ('selector', ItemSelector(key='string')),
                            ('homophones for pun', Homophone_0_1()),
                        ('vect', DictVectorizer())
                        ]))
                        ,
                        ('idiom 0 1', Pipeline([
                            ('selector', ItemSelector(key='string')),
                            ('idiom for pun', Idioms()),
                        ('vect', DictVectorizer())
                        ])),                        
                        ('antonyms 0 1', Pipeline([
                            ('selector', ItemSelector(key='string')),
                            ('antonyms for pun', Antonyms()),
                        ('vect', DictVectorizer())
                        ])),                      
                        ('homonym 0 1', Pipeline([
                            ('selector', ItemSelector(key='string')),
                            ('homonym for pun', Homonym()),
                        ('vect', DictVectorizer())
                        ]))
                        
                        
                        
                    ]
                )
            ),
            (
                # TODO: Which classifier should be used here? Also probably
                # want to do cross-validation on hyperparameters.
                "clf",
                SGDClassifier(loss='log', penalty='l2', alpha=self.alpha, max_iter=15000, shuffle=True)
            )
        ])

    def train(self, x_train, y_train):
        self.pipeline.fit(x_train, y_train)
        return self.pipeline.predict(x_train)

    # Make predictions on test data. Use the y_test labels to find accuracy of predictions.
    def test(self, x_test):
        return self.pipeline.predict(x_test)

    def test_with_probabilities(self, x_test):
        return self.pipeline.predict_proba(x_test)

    def fit(self, x_train, y_train=None):

        self.train(x_train, y_train)

        return self


    def predict(self, x):
        return self.test(x)

    def score(self, x, y, sample_weight=None):
        """Returns the mean accuracy on the given test data and labels.

        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.

        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True labels for X.

        sample_weight : array-like, shape = [n_samples], optional
            Sample weights.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.

        """
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(x), sample_weight=sample_weight)
