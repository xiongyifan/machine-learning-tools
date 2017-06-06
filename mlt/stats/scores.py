"""scores, for example: accuracy, error rate"""

import numpy as np


def cal_accuracy(y_predict, y):
    """cal accuracy

    Parameters
    ----------
    y_predict : ndarray
    y : ndarray
    """
    return np.average(y_predict == y)


def cal_error_rate(y_predict, y):
    """cal error rate"""
    return 1.0 - cal_accuracy(y_predict, y)

# todo: later I will change this function
# def verification(x_train, y_train, x_test, y_test, w):
#     """verification the difference from train and test set"""
#     accuracy_train = cal_accuracy(x_train, y_train, w)
#     accuracy_test = cal_accuracy(x_test, y_test, w)
#     logger.info('train accuracy : ', accuracy_train, '; tests accuracy : ', accuracy_test, '; difference : ',
#                 (accuracy_train - accuracy_test))
