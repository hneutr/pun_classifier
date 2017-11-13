from pun_data import HeterographicData, HomographicData
from baseline_location import BaselinePunLocationClassifier
from baseline_detection import BaselinePunDetectionClassifier
from pun_detection_with_features import PunDetectionWithFeaturesClassifier
from eval import Eval
import argparse

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
    args = parser.parse_args()

    if args.graphic == "homographic":
        data = HomographicData()
    elif args.graphic =="heterographic":
        data = HeterographicData()
    else :
        raise Exception('Invalid pun type specified.')

    # PUN DETECTION
    if args.detection:
        # Create baseline pun detection classifier, train it, and get predictions on test data
        if args.baselines:
            baselineDetectionClassifier = BaselinePunDetectionClassifier()
            baselineDetectionClassifier.train(data.x_train, data.y_train)
            baselineDetectionPredicted = baselineDetectionClassifier.test(data.x_test, data.y_test)

            # Evaluate baseline classifier
            Eval.evaluateDetection(baselineDetectionPredicted, data.y_test)

        # Create pun detection classifier, train it, and get predictions on test data
        print("Running pun detection classifier with features...")
        punDetectionClassifier = PunDetectionWithFeaturesClassifier()
        punDetectionClassifier.train(data.x_train, data.y_train)
        detectionPredicted = punDetectionClassifier.test(data.x_test, data.y_test)

        # Evaluate pun detection classifier
        confusion_matrix = Eval.confusion_matrix(detectionPredicted, data.y_test)
        Eval.plot_precision_recall(detectionPredicted, data.y_test)



    # PUN LOCATION


    # Create baseline pun detection classifier, train it, and get predictions on test data
    baselineLocationClassifier = BaselinePunLocationClassifier()
    baselineLocationClassifier.train(data.x_train, data.y_train)
    baselineLocationPredicted = baselineLocationClassifier.test(data.x_test)

    # Evaluate baseline classifier
    # Eval.evaluateDetection(baselineLocationPredicted, data.y_test)

    # TODO
    # Pun locater would go here, but likely won't start that until pun detection classifier is working
