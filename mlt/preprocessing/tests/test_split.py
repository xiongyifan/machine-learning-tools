"""test for null_value.py"""

import unittest

from mlt.file import read
from mlt.preprocessing import split


# noinspection PyMethodMayBeStatic
class SplitTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_separate_x_y(self):
        df = read.from_csv('../../../data/adult.csv', ['#NAME?'])
        X, y = split.separate_x_y(df, 'income')
        self.assertEqual(X.shape, (5000, 14), 'assert x shape')
        self.assertEqual(y.shape, (5000,), 'assert y shape')

    def test_split_train_val_test(self):
        df = read.from_csv('../../../data/adult.csv', ['#NAME?'])

        train, val, test = split.split_train_val_test(df, val_percentage=0.2, test_percentage=0.2)
        self.assertEqual(train.shape, (3000, 15))
        self.assertEqual(val.shape, (1000, 15))
        self.assertEqual(test.shape, (1000, 15))

        X, y = split.separate_x_y(df, 'income')
        X_train, y_train, X_val, y_val, X_test, y_test = split.split_train_val_test(X, y, val_percentage=0.1,
                                                                                    test_percentage=0.1)
        self.assertEqual(X_train.shape, (4000, 14))
        self.assertEqual(y_train.shape, (4000,))
        self.assertEqual(X_val.shape, (500, 14))
        self.assertEqual(y_val.shape, (500,))
        self.assertEqual(X_test.shape, (500, 14))
        self.assertEqual(y_test.shape, (500,))

    def test_split_data(self):
        df = read.from_csv('../../../data/adult.csv', ['#NAME?'])

        train, val = split.split_data(df, percentage=0.2)
        self.assertEqual(train.shape, (4000, 15))
        self.assertEqual(val.shape, (1000, 15))
