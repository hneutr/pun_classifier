import argparse

from baseline_detection import BaselinePunDetectionClassifier
from baseline_location import BaselinePunLocationClassifier
from eval import Eval
from pun_data import Data
from pun_detection_with_features import PunDetectionWithFeaturesClassifier


def printCurrentClassifier(name):
    print("\n\n---- Running  ", name, " Classifier --------\n")

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
    parser.add_argument('--even', action="store_false", default=True,
                        help="run the algorithms on the more evenly split dataset")
    args = parser.parse_args()

    data = Data(args.graphic, args.even)

    # PUN DETECTION
    if args.detection:
        # Create baseline pun detection classifier, train it, and get predictions on test data
        if args.baselines:
            printCurrentClassifier("Baseline Detection")
            baselineDetectionClassifier = BaselinePunDetectionClassifier()
            baselineDetectionTrainingPredicted = baselineDetectionClassifier.train(data.x_train, data.y_train)
            Eval.evaluateAccuracy(baselineDetectionTrainingPredicted, data.y_train, 'training')
            baselineDetectionPredicted = baselineDetectionClassifier.test(data.x_test, data.y_test)

            # Evaluate baseline classifier
            Eval.evaluateDetection(baselineDetectionPredicted, data.y_test)

        # Create pun detection classifier, train it, and get predictions on test data
        printCurrentClassifier("Pun Detection With Features")
        punDetectionClassifier = PunDetectionWithFeaturesClassifier()
        detectionTrainingPredicted = punDetectionClassifier.train(data.x_train, data.y_train)
        Eval.evaluateAccuracy(detectionTrainingPredicted, data.y_train, 'training')
        detectionPredicted = punDetectionClassifier.test(data.x_test, data.y_test)

        # Evaluate pun detection classifier
        Eval.evaluateDetection(detectionPredicted, data.y_test)





    # PUN LOCATION

    printCurrentClassifier("Baseline Location")
    # Create baseline pun detection classifier, train it, and get predictions on test data
    baselineLocationClassifier = BaselinePunLocationClassifier()
    baselineLocationClassifier.train(data.x_train, data.y_train)
    baselineLocationPredicted = baselineLocationClassifier.test(data.x_test)

    # Evaluate baseline classifier
    # Eval.evaluateDetection(baselineLocationPredicted, data.y_test)

    # TODO
    # Pun locater would go here, but likely won't start that until pun detection classifier is working
