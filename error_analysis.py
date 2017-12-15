import pickle
import matplotlib
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

class ErrorAnalysis:
    def __init__(self, xs, labels, predictions, task, graphic, classifier):
        self.xs = xs
        self.labels = labels
        self.predictions = predictions
        self.task = task
        self.graphic = graphic
        self.classifier = classifier

        self.analyze_sentence_length()
    
    def analyze_sentence_length(self):
        totals = defaultdict(int)
        corrects = defaultdict(int)
        for x, true, pred in zip(self.xs, self.labels, self.predictions):
            totals[len(x)] += 1

            if pred == true:
                corrects[len(x)] += 1

        lengths = sorted(totals.keys())
        correct_counts = [ corrects[l] for l in lengths ]
        total_counts = [ totals[l] for l in lengths ]

        print(lengths)

        percents = []
        for c, t in zip(correct_counts, total_counts):
            percent = c/t if c else 0.0
            percents.append(percent)

        plt.clf()
        plt.scatter(lengths, percents)

        plt.title("Accuracy by sentence length")
        plt.xlabel("sentence length")
        plt.ylabel("accuracy")

        path = 'error_analysis/sentence_length/%s-%s-%s' % (self.task, self.graphic, self.classifier)
        plt.savefig(path)

        with open("%s.pkl" % path, "wb") as f:
            info = [total_counts, correct_counts, percents]
            pickle.dump(info, f)

def get_data(graphic, algorithm):
    path = 'error_analysis/sentence_length/location-%s-%s.pkl' % (graphic, algorithm)

    with open(path, 'rb') as f:
        _, _, percents = pickle.load(f)

    return percents

def graph_it(algorithm):
    hom = get_data('homographic', algorithm)
    hom_lens = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 33, 34, 37, 48, 50]

    het = get_data('heterographic', algorithm)
    het_lens = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36 , 40]

    plt.clf()
    plt.scatter(hom_lens, hom, c='red', label="Homographic")
    plt.scatter(het_lens, het, c='green', label="Heterographic")
    plt.xlabel("sentence length")
    plt.ylabel("accuracy")
    plt.legend()
    path = 'error_analysis/sentence_length/location-%s' % algorithm
    plt.savefig(path)

if __name__ == "__main__":
    graph_it('window')
    graph_it('rnn')
