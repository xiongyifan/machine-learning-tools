import numpy as np
import random


def get_data_hw1_15():
    x_y_train = np.fromfile('../data/hw1_15_train.dat', dtype=float, sep=' ').reshape(-1, 5)
    # print x_y_train
    print('x_y_train.shape', x_y_train.shape)

    x_train = x_y_train[:, :4]
    # print x_train
    print('x_train.shape', x_train.shape)

    x0 = np.ones(x_train.shape[0], dtype=float).reshape(-1, 1)
    # print x0
    print('x0.shape', x0.shape)

    x_train = np.hstack((x0, x_train))
    # print x_train
    print('x_train.shape', x_train.shape)

    y_train = x_y_train[:, 4]
    # print y
    print('y.shape', y_train.shape)

    m, n = x_train.shape

    return x_train, y_train, m, n


def get_data_hw1_18():
    x_y_train = np.fromfile('../data/hw1_18_train.dat', dtype=float, sep=' ').reshape(-1, 5)
    x_y_test = np.fromfile('../data/hw1_18_test.dat', dtype=float, sep=' ').reshape(-1, 5)
    # print X_y
    # print('X_y.shape', x_y_train.shape

    x_train = x_y_train[:, :4]
    x_test = x_y_test[:, :4]
    # print X
    # print('X.shape', x_train.shape

    x0 = np.ones(x_train.shape[0], dtype=float).reshape(-1, 1)
    # print x0
    # print('x0.shape', x0.shape

    x_train = np.hstack((x0, x_train))
    x_test = np.hstack((x0, x_test))
    # print X
    # print('X.shape', x_train.shape

    y_train = x_y_train[:, 4]
    y_test = x_y_test[:, 4]
    # print y
    # print('y.shape', y_train.shape

    m, n = x_train.shape

    return x_train, y_train, x_test, y_test, m, n


def cal_accuracy(X, Y, w):
    return np.average(np.sign(X.dot(w)) == Y)


def cal_error_rate(X, Y, w):
    return 1.0 - cal_accuracy(X, Y, w)


def get_order(order, m):
    if order is 'sequence':
        order = list(range(m))
    elif order is 'random':
        order = list(range(m))
        random.shuffle(order)

    return order


def init_w(n):
    w = np.zeros(n, dtype=float)
    return w


def verification(x_train, y_train, x_test, y_test, w):
    accuracy_train = cal_accuracy(x_train, y_train, w)
    accuracy_test = cal_accuracy(x_test, y_test, w)
    print('train accuracy : ', accuracy_train, '; tests accuracy : ', accuracy_test, '; difference : ', (accuracy_train - accuracy_test))


def add_ploy(X, n, is_skip_x0):
    if is_skip_x0:
        X_powered = np.power(X[:, 1:], n)
    else:
        X_powered = np.power(X, n)
    X = np.hstack((X, X_powered))
    return X