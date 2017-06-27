"""test for read.py"""

import unittest

import numpy as np

from mlt.file import read
from mlt.file import save
from mlt.preprocessing import split


# noinspection PyMethodMayBeStatic
class SaveTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_to_npz(self):
        data_name = 'mnist'
        data_path = '../../../data/' + data_name + '.csv'
        npz_path = '../../../data/' + data_name + '.npz'

        df = read.from_csv(data_path)
        # df = read.from_csv(data_path, '#NAME?')

        X, y = split.separate_x_y(df, 'label')
        # X, y = split.separate_x_y(df, 'income')
        X_train, y_train, X_val, y_val, X_test, y_test = split.split_train_val_test(X, y, val_percentage=0.1,
                                                                                    test_percentage=0.1)

        save.train_val_test_to_npz(npz_path, X_train, y_train, X_val, y_val, X_test, y_test)
        X_train_2, y_train_2, X_val_2, y_val_2, X_test_2, y_test_2 = read.from_npz_train_val_test_x_y(
            npz_path)

        np.testing.assert_array_equal(X_train, X_train_2)
