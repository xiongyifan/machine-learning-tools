import numpy as np


def cal_accuracy(X, Y, w):
    return np.average(np.sign(X.dot(w)) == Y)


def cal_error_rate(X, Y, w):
    return 1.0 - cal_accuracy(X, Y, w)


def verification(x_train, y_train, x_test, y_test, w):
    accuracy_train = cal_accuracy(x_train, y_train, w)
    accuracy_test = cal_accuracy(x_test, y_test, w)
    print('train accuracy : ', accuracy_train, '; tests accuracy : ', accuracy_test, '; difference : ', (accuracy_train - accuracy_test))
