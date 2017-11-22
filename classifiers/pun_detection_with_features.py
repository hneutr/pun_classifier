from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction import DictVectorizer

from features.item_selector import ItemSelector
from features.lesk_algorithm_transformer import LeskAlgorithmTransformer
from features.raw_transformer import RawTransformer
from features.pos_transformer import PosTransformer
from word_embeddings import WordEmbeddings

from features.n_grams import Unigram, Bigrams, Trigrams
from features.negative_positive import Negatives, Positives
from features.case_sensetives import AllUpper
from features.homophones import Homophone



# Pun detection classifier using feature engineering
class PunDetectionWithFeaturesClassifier:
    def __init__(self):
        self.name = "Pun Detection With Features"
        self.raw_embeddings = WordEmbeddings()

        self.pipeline = Pipeline([
            (
                "raw",
                RawTransformer({
                    'tokens': lambda x: x,
                    'string': lambda x: " ".join(x),
                    'embeddings': lambda x: self.raw_embeddings.embed(x),
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
                        
                        
                        
                        
                        
                        
                        
                    ]
                )
            ),
            (
                # TODO: Which classifier should be used here? Also probably
                # want to do cross-validation on hyperparameters.
                "sgd",
                SGDClassifier(loss='log', penalty='l2', alpha=0.0001, max_iter=15000, shuffle=True)
            )
        ])

    def train(self, x_train, y_train):
        self.pipeline.fit(x_train, y_train)
        return self.pipeline.predict(x_train)

    # Make predictions on test data. Use the y_test labels to find accuracy of predictions.
    def test(self, x_test):
        return self.pipeline.predict(x_test)
