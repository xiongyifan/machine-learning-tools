"""scores, for example: accuracy, error rate"""

import numpy as np

from mlt.utils.logging import logger


def cal_accuracy(X, y, w):
    """cal accuracy return percentage"""
    return np.average(np.sign(X.dot(w)) == y)


def cal_error_rate(X, Y, w):
    """cal error rate return percentage"""
    return 1.0 - cal_accuracy(X, Y, w)


def verification(x_train, y_train, x_test, y_test, w):
    """verification the difference from train and test set"""
    accuracy_train = cal_accuracy(x_train, y_train, w)
    accuracy_test = cal_accuracy(x_test, y_test, w)
    logger.info('train accuracy : ', accuracy_train, '; tests accuracy : ', accuracy_test, '; difference : ',
                (accuracy_train - accuracy_test))
