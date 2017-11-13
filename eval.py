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

    @staticmethod
    def plot_precision_recall(y_pred, y_true):
        # All this code has been taken from http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
        average_precision = average_precision_score(y_true, y_pred)

        print('Average precision-recall score: {0:0.2f}'.format(
            average_precision))

        precision, recall, _ = precision_recall_curve(y_true, y_pred)

        plt.step(recall, precision, color='b', alpha=0.2,
                 where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2,
                         color='b')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))

        plt.show()


