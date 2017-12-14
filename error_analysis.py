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

