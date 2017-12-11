# TODO - Need cross validation of hyperparameters used in classifiers
# and in some files choice of classifier used
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import argparse

# Utility function to report best scores
from classifiers.pun_detection_with_features import PunDetectionWithFeaturesClassifier
from classifiers.scikit_wrapper import ScikitWrapperClassifier
from pun_data import DetectionData

# Report function has been taken from the example at
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html#sphx-glr-auto-examples-model-selection-plot-randomized-search-py
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

def run_cross_validation(clf, param_grid, data):
    grid_search = GridSearchCV(clf, param_grid=param_grid)
    grid_search.fit(data.x_train, data.y_train)

    report(grid_search.cv_results_)

# Note - this is still a rough skeleton and not everything is being cross validated
# This is just a start to this file
if __name__ == "__main__":

    # Note - not all these arguments currently needed/used but here for the future when we cross validate more things :)
    parser = argparse.ArgumentParser(description='type of pun')
    parser.add_argument('--graphic', type=str, default='homographic',
                        help="which type of pun ['homographic', 'heterographic']. Default: homographic")
    parser.add_argument('--baselines', action="store_true", default=False,
                        help="run baselines or not. defaults to false.")
    parser.add_argument('--detection', action="store_true", default=False,
                        help="run detection algorithms or not. defaults to false.")
    parser.add_argument('--location', action="store_true", default=False,
                        help="run location algorithms or not. defaults to false.")
    parser.add_argument('--window', action="store_true", default=False,
                        help="run the sliding window algorithm for location")
    parser.add_argument('--rnn', action="store_true", default=False,
                        help="run the rnn algorithm")
    parser.add_argument('--features', action="store_true", default=False,
                        help="run the sgd algorithm for detection")
    parser.add_argument('--even', action="store_false", default=True,
                        help="run the algorithms on the more evenly split dataset")
    parser.add_argument('--use_cached', action="store_true", default=False,
                        help="use cached models if available")
    parser.add_argument('--ensemble', action="store_true", default=False,
                        help="use voting classifier")
    parser.add_argument('--adaboost', action="store_true", default=False,
                        help="use adaboost for sliding window classifier")

    args = parser.parse_args()

    if args.detection:
        detectionData = DetectionData(args.graphic, args.even)

        if args.features:
            # Specify classifier
            clf = PunDetectionWithFeaturesClassifier()

            # specify parameters and distributions to sample from
            param_grid = {"alpha": [0.1, 0.01, 0.001]}

            # run grid search cross validation
            run_cross_validation(clf, param_grid, detectionData)


