import sys
import logging
import pandas as pd
import numpy as np

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)


# class Perceptron:
#     def fit(self, x, y, n):
#         cols = len(x)
#         rows = len(y[4].unique())
#         weights = np.zeros(rows, cols)
#         x = x.to_numpy()
#         y = y.to_numpy()
#
#         for i in range(cols):
#             logging.debug("a")

def fit(x, t):
    x = x.to_numpy()
    t = t.to_numpy()
    w = np.zeros((4, 1))

    y = perceptron_output(w, 1, x)
    t_y = t - y
    dw = n * np.dot(x.T, t_y)
    w = w + dw

    return w


def step_function(x):
    return 1 if x >= 0 else 0


def perceptron_output(weights, bias, x):
    # calculation = np.dot(x, weights) + bias
    calculation = np.dot(x, weights)
    return calculation
    # return step_function(calculation)


iris_2_data = pd.read_csv("./data/iris_2_classes.txt", usecols=[0, 1, 2, 3], header=None)
iris_2_label = pd.read_csv("./data/iris_2_classes.txt", usecols=[4], header=None)
classes = {'setosa': 1, 'versicolor': -1}
iris_2_label[4] = [classes[item] for item in iris_2_label[4]]

n = 1
result = fit(iris_2_data, iris_2_label)
logging.debug(result)
