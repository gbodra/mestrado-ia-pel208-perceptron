import sys
import logging
import pandas as pd
import numpy as np

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)


class Perceptron:
    def fit(self, x, y, n):
        cols = len(x)
        rows = len(y[4].unique())
        weights = np.zeros(rows, cols)
        x = x.to_numpy()
        y = y.to_numpy()

        for i in range(cols):
            logging.debug("a")


iris_2_data = pd.read_csv("./data/iris_2_classes.txt", usecols=[0, 1, 2, 3], header=None)
iris_2_label = pd.read_csv("./data/iris_2_classes.txt", usecols=[4], header=None)
labels = {'setosa': -1, 'versicolor': 1}
p = Perceptron()
p.fit(iris_2_data, iris_2_label, 1)
