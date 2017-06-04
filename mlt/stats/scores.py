import numpy as np


def cal_accuracy(X, Y, w):
    return np.average(np.sign(X.dot(w)) == Y)


def cal_error_rate(X, Y, w):
    return 1.0 - cal_accuracy(X, Y, w)
