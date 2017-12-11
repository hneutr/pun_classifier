import argparse

from cache.cache import Cache
from classifiers.adaboost_sliding_window import AdaboostSlidingWindowClassifier
from classifiers.baseline import BaselinePunClassifier
from classifiers.pun_detection_with_features import PunDetectionWithFeaturesClassifier
from classifiers.pun_rnn import PunRNNClassifier
from classifiers.pun_rnn_detection import PunRNNDetectionClassifier
from classifiers.pun_location_with_features import PunLocationWithFeaturesClassifier
from classifiers.scikit_wrapper import ScikitWrapperClassifier
from classifiers.sliding_window import PunSlidingWindowClassifier
from classifiers.voting_classifier import PunVotingClassifier
from eval import Eval
from pun_data import DetectionData, LocationData

cache = Cache()

def runClassifier(classifier, data, evalFn, useCache):
    print("\n\n---- Running  ", classifier.name, " Classifier --------\n")

    # Use cached version of model if possible, otherwise train it
    if useCache and cache.has(classifier):
        classifier = cache.get(classifier)
        print("Using cached version of classifier")
    else:
        print("Training classifier...")
        # Train the classifier and evaluate it's training accuracy
        trainingPredicted = classifier.train(data.x_train, data.y_train)
        Eval.evaluateAccuracy(trainingPredicted, data.y_train, 'training')
        cache.set(classifier)

    print("Testing classifier...")
    # Test the classifier to get predictions
    y_pred = classifier.test(data.x_test)

    # Evaluate classifier on predicted output
    evalFn(classifier.name, y_pred, data.y_test)


if __name__ == "__main__":
    # Get pun data for training and for testing
    data = None

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
    parser.add_argument('--sgd', action="store_true", default=False,
                        help="run the sgd algorithm for detection")
    parser.add_argument('--decision_tree', action = "store_true", default = False,
                        help="run the decision_tree classifier for location")
    parser.add_argument('--even', action="store_false", default=True,
                        help="run the algorithms on the more evenly split dataset")
    parser.add_argument('--use_cached', action="store_true", default=False,
                        help="use cached models if available")
    parser.add_argument('--ensemble', action="store_true", default=False,
                        help="use voting classifier")
    parser.add_argument('--adaboost', action="store_true", default=False,
                        help="use adaboost for sliding window classifier")

    args = parser.parse_args()

    print("Running %s puns" % args.graphic)

    # PUN DETECTION
    if args.detection:

        detectionData = DetectionData(args.graphic, args.even)
        baselinePunClassifier = BaselinePunClassifier(type="Detection")
        punRnnDetectionClassifier = PunRNNDetectionClassifier()
        punDetectionWithFeaturesClassifier = PunDetectionWithFeaturesClassifier()

        # Create baseline pun detection classifier
        if args.baselines:
            runClassifier(baselinePunClassifier, detectionData, Eval.evaluateDetection, args.use_cached)

        if args.rnn:
            runClassifier(punRnnDetectionClassifier, detectionData, Eval.evaluateDetection, args.use_cached)

        if args.sgd:
            runClassifier(punDetectionWithFeaturesClassifier, detectionData, Eval.evaluateDetection, args.use_cached)

        if args.ensemble:
            classifiers = [
                ScikitWrapperClassifier(baselinePunClassifier),
                ScikitWrapperClassifier(punRnnDetectionClassifier),
                ScikitWrapperClassifier(punDetectionWithFeaturesClassifier)
            ]
            runClassifier(PunVotingClassifier(type="Detection", classifiers=classifiers), detectionData, Eval.evaluateDetection, args.use_cached)


    # PUN LOCATION
    if args.location:

        locationData = LocationData(args.graphic)
        baselinePunLocationClassifier = BaselinePunClassifier(type="Location")
        punRnnLocationClassifier = PunRNNClassifier(output="word")
        punDecisionTreeClassifier = PunLocationWithFeaturesClassifier(output = "word")
        punSlidingWindowClassifier = PunSlidingWindowClassifier(output="word")
        adaboostSlidingWindowClassifier = AdaboostSlidingWindowClassifier()

        # Create baseline pun location classifier
        if args.baselines:
            runClassifier(baselinePunLocationClassifier, locationData, Eval.evaluateLocation, args.use_cached)

        if args.rnn:
            runClassifier(punRnnLocationClassifier, locationData, Eval.evaluateLocation, args.use_cached)

        if args.decision_tree:
            runClassifier(punDecisionTreeClassifier, locationData, Eval.evaluateLocation, args.use_cached)

        if args.window:
            runClassifier(punSlidingWindowClassifier, locationData, Eval.evaluateLocation, args.use_cached)

        if args.adaboost:
            runClassifier(adaboostSlidingWindowClassifier, locationData, Eval.evaluateLocation, args.use_cached)

        if args.ensemble:
            classifiers = [
                ScikitWrapperClassifier(baselinePunLocationClassifier),
                ScikitWrapperClassifier(punRnnLocationClassifier),
                ScikitWrapperClassifier(punDecisionTreeClassifier),
                ScikitWrapperClassifier(punSlidingWindowClassifier),
                ScikitWrapperClassifier(adaboostSlidingWindowClassifier)
            ]
            runClassifier(PunVotingClassifier(type="Location", classifiers=classifiers), locationData, Eval.evaluateLocation, args.use_cached)


    # Output final report
    Eval.print_reports()
