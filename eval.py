# Evaluation for pun detection classifiers
# Might compute accuracy, precision, f-measure, etc...

from pandas_ml import ConfusionMatrix

# TODO
class Eval:

    @staticmethod
    def evaluateDetection(y_pred, y_true):
        pass

    @staticmethod
    def evaluateLocation(y_pred, y_true):
        pass

    @staticmethod
    def confusion_matrix(y_pred, y_true):
        confusion_matrix = ConfusionMatrix(y_true, y_pred)
        print("Confusion matrix:\n%s" % confusion_matrix)


