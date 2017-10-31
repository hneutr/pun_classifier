from pun_data import Data
from baseline_detection import BaselinePunDetectionClassifier
from baseline_location import BaselinePunLocationClassifier
from pun_detection import PunDetectionClassifier
from eval import Eval

if __name__ == "__main__":
    # Get pun data for training and for testing
    data = Data()
    x_train, y_train = data.x_train, data.y_train
    x_test, y_test = data.x_test, data.y_test


    # PUN DETECTION

    # Create baseline pun detection classifier, train it, and get predictions on test data
    baselineDetectionClassifier = BaselinePunDetectionClassifier()
    baselineDetectionClassifier.train(x_train, y_train)
    baselineDetectionPredicted = baselineDetectionClassifier.test(x_test)

    # Evaluate baseline classifier
    Eval.evaluateDetection(baselineDetectionPredicted, y_test)

    # Create pun detection classifier, train it, and get predictions on test data
    punDetectionClassifier = PunDetectionClassifier()
    punDetectionClassifier.train(x_train, y_train)
    detectionPredicted = punDetectionClassifier.test(x_test)

    # Evaluate pun detection classifier
    Eval.evaluateDetection(detectionPredicted, y_test)



    # PUN LOCATION


    # Create baseline pun detection classifier, train it, and get predictions on test data
    baselineLocationClassifier = BaselinePunLocationClassifier()
    baselineLocationClassifier.train(x_train, y_train)
    baselineLocationPredicted = baselineLocationClassifier.test(x_test)

    # Evaluate baseline classifier
    Eval.evaluateLocation(baselineLocationPredicted, y_test)

    # TODO
    # Pun locater would go here, but likely won't start that until pun detection classifier is working