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
from error_analysis import ErrorAnalysis

cache = Cache()

def runClassifier(classifier, data, evalFn, runParams):
    print("\n\n---- Running  ", classifier.name, " Classifier --------\n")

    # Use cached version of model if possible, otherwise train it
    if runParams['cache'] and cache.has(classifier):
        classifier = cache.get(classifier)
        print("Using cached version of classifier")
    else:
        print("Training classifier...")
        # Train the classifier and evaluate it's training accuracy
        trainingPredicted = classifier.train(data.x_train, data.y_train)
        Eval.evaluateAccuracy(trainingPredicted, data.y_train, 'training')
        # cache.set(classifier)

    print("Testing classifier...")
    # Test the classifier to get predictions
    y_pred = classifier.test(data.x_test)

    # Evaluate classifier on predicted output
    evalFn(classifier.name, y_pred, data.y_test)

    if runParams['error_analysis']:
        ErrorAnalysis(data.x_test, data.y_test, y_pred, runParams['task'], runParams['graphic'], runParams['classifier'])

        



if __name__ == "__main__":
    # Get pun data for training and for testing
    data = None

    parser = argparse.ArgumentParser(description='type of pun')
    parser.add_argument('--graphic', type=str, default='homographic',
                        help="which type of pun ['homographic', 'heterographic', 'combined', 'both']. Default: homographic")
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
    parser.add_argument('--error_analysis', action="store_true", default=False,
                        help="analyze the errors")
    args = parser.parse_args()

    print("Running %s puns" % args.graphic)

    run_params = {
        'graphic' : args.graphic,
        'cache' : args.use_cached,
        'error_analysis' : args.error_analysis,
    }

    # PUN DETECTION
    if args.detection:
        run_params['task'] = 'detection'

        detectionData = DetectionData(args.graphic, args.even)

        baselinePunClassifier = BaselinePunClassifier(type="Detection")
        punRnnDetectionClassifier = PunRNNDetectionClassifier(args.graphic)
        punDetectionWithFeaturesClassifier = PunDetectionWithFeaturesClassifier()

        # Create baseline pun detection classifier
        if args.baselines:
            run_params['classifier'] = 'baseline'
            runClassifier(baselinePunClassifier, detectionData, Eval.evaluateDetection, run_params)

        if args.rnn:
            run_params['classifier'] = 'rnn'
            runClassifier(punRnnDetectionClassifier, detectionData, Eval.evaluateDetection, run_params)

        if args.features:
            run_params['classifier'] = 'features'
            runClassifier(punDetectionWithFeaturesClassifier, detectionData, Eval.evaluateDetection, run_params)

        if args.ensemble:
            run_params['classifier'] = 'ensemble'
            classifiers = [
                baselinePunClassifier,
                punRnnDetectionClassifier,
                punDetectionWithFeaturesClassifier
            ]
            runClassifier(PunVotingClassifier(type="Detection", classifiers=classifiers), detectionData, Eval.evaluateDetection, run_params)


    # PUN LOCATION
    if args.location:
        run_params['task'] = 'location'

        locationData = LocationData(args.graphic)
        baselinePunLocationClassifier = BaselinePunClassifier(type="Location")
        punRnnLocationClassifier = PunRNNClassifier(output="word")
        punDecisionTreeClassifier = PunLocationWithFeaturesClassifier(output = "word")
        punSlidingWindowClassifier = PunSlidingWindowClassifier(output="word")
        adaboostSlidingWindowClassifier = AdaboostSlidingWindowClassifier()

        # Create baseline pun location classifier
        if args.baselines:
            run_params['classifier'] = 'baseline'
            runClassifier(baselinePunLocationClassifier, locationData, Eval.evaluateLocation, run_params)

        if args.rnn:
            run_params['classifier'] = 'rnn'
            runClassifier(punRnnLocationClassifier, locationData, Eval.evaluateLocation, run_params)

        if args.decision_tree:
            run_params['classifier'] = 'decision_tree'
            runClassifier(punDecisionTreeClassifier, locationData, Eval.evaluateLocation, run_params)

        if args.window:
            run_params['classifier'] = 'window'
            runClassifier(punSlidingWindowClassifier, locationData, Eval.evaluateLocation, run_params)

        if args.adaboost:
            run_params['classifier'] = 'adaboost'
            runClassifier(adaboostSlidingWindowClassifier, locationData, Eval.evaluateLocation, run_params)

        if args.ensemble:
            run_params['classifier'] = 'ensemble'
            classifiers = [
                baselinePunLocationClassifier,
                punRnnLocationClassifier,
                punDecisionTreeClassifier,
                punSlidingWindowClassifier,
                # ScikitWrapperClassifier(adaboostSlidingWindowClassifier)  - adaboost not yet working
            ]
            runClassifier(PunVotingClassifier(type="Location", classifiers=classifiers), locationData, Eval.evaluateLocation, run_params)


    # Output final report
    Eval.print_reports()
