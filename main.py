import argparse

from classifiers.baseline import BaselinePunClassifier
from classifiers.pun_rnn import PunRNNClassifier
from classifiers.pun_detection_with_features import PunDetectionWithFeaturesClassifier
from classifiers.sliding_window import PunSlidingWindowClassifier

from eval import Eval
from pun_data import DetectionData, LocationData


def runClassifier(classifier, data, evalFn):
    print("\n\n---- Running  ", classifier.name, " Classifier --------\n")

    # Train the classifier and evaluate it's training accuracy
    trainingPredicted = classifier.train(data.x_train, data.y_train)
    # Eval.evaluateAccuracy(trainingPredicted, data.y_train, 'training')
    evalFn(classifier.name, trainingPredicted, data.y_train)

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
    parser.add_argument('--even', action="store_false", default=True,
                        help="run the algorithms on the more evenly split dataset")
    args = parser.parse_args()

    print("Running %s puns" % args.graphic)


    # PUN DETECTION
    if args.detection:

        detectionData = DetectionData(args.graphic, args.even)

        # Create baseline pun detection classifier
        if args.baselines:
            runClassifier(BaselinePunClassifier(), detectionData, Eval.evaluateDetection)

        runClassifier(PunDetectionWithFeaturesClassifier(), detectionData, Eval.evaluateDetection)
        runClassifier(PunRNNClassifier(), detectionData, Eval.evaluateDetection)


    # PUN LOCATION
    if args.location:

        locationData = LocationData(args.graphic)

        # Create baseline pun location classifier
        if args.baselines:
            runClassifier(BaselinePunClassifier(), locationData, Eval.evaluateLocation)

        runClassifier(PunRNNClassifier(), locationData, Eval.evaluateLocation)
        runClassifier(PunSlidingWindowClassifier(), locationData, Eval.evaluateLocation)


    # Output final report
    Eval.print_reports()
