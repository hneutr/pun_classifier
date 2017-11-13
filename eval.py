# Evaluation for pun detection classifiers
# Might compute accuracy, precision, f-measure, etc...

from pandas_ml import ConfusionMatrix
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score

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




