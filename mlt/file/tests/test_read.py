"""test for read.py"""

import unittest

import numpy as np

from mlt.file import read


class ReadTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_from_csv(self):
        df = read.from_csv('../../../data/adult.csv', ['#NAME?'])

        # 1. check the data shape
        self.assertEqual(df.shape, (5000, 15), 'assert the data shape')

        # 2. check the null value
        self.assertTrue(np.isnan(df.iloc[16]['age']), 'assert the null value')
