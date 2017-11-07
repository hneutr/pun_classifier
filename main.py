from pun_data import HeterographicData
from baseline_detection import BaselinePunDetectionClassifier
from baseline_location import BaselinePunLocationClassifier
from pun_detection_with_features import PunDetectionWithFeaturesClassifier
from eval import Eval
import pickle

if __name__ == "__main__":
    # Get pun data for training and for testing

    data = HeterographicData()

    # PUN DETECTION

    # Create baseline pun detection classifier, train it, and get predictions on test data
    baselineDetectionClassifier = BaselinePunDetectionClassifier()
    baselineDetectionClassifier.train(data.x_train, data.y_train)
    baselineDetectionPredicted = baselineDetectionClassifier.test(data.x_test)

    # Evaluate baseline classifier
    Eval.evaluateDetection(baselineDetectionPredicted, data.y_test)

    # Create pun detection classifier, train it, and get predictions on test data
    punDetectionClassifier = PunDetectionWithFeaturesClassifier()
    punDetectionClassifier.train(data.x_train, data.y_train)
    detectionPredicted = punDetectionClassifier.test(data.x_test, data.y_test)

    # Evaluate pun detection classifier
    Eval.evaluateDetection(detectionPredicted, data.y_test)





    # PUN LOCATION


    # Create baseline pun detection classifier, train it, and get predictions on test data
    baselineLocationClassifier = BaselinePunLocationClassifier()
    baselineLocationClassifier.train(data.x_train, data.y_train)
    baselineLocationPredicted = baselineLocationClassifier.test(data.x_test)

    # Evaluate baseline classifier
    Eval.evaluateLocation(baselineLocationPredicted, data.y_test)

    # TODO
    # Pun locater would go here, but likely won't start that until pun detection classifier is working