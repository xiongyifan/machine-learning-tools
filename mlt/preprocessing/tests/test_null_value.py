"""test for null_value.py"""

import unittest

import pandas as pd
import numpy as np
from pandas.util.testing import assert_frame_equal

from mlt.preprocessing import null_value


# noinspection PyMethodMayBeStatic
class NullValueTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_fill_median(self):
        data = pd.DataFrame(np.array([np.nan, 10.0, 11.0]), columns=['age'])
        expected = pd.DataFrame(np.array([10.5, 10.0, 11.0]), columns=['age'])
        result = null_value.fill_median(data, 'age')
        assert_frame_equal(expected, result)

    def test_remove_rows_by_column(self):
        data = pd.DataFrame(np.array([np.array([np.nan, 60.0]), np.array([10.0, np.nan]), np.array([11.0, 80.0])]),
                            columns=['age', 'score'])
        expected = pd.DataFrame(np.array([np.array([np.nan, 60.0]), np.array([11.0, 80.0])]), columns=['age', 'score'])
        result = null_value.remove_rows_by_column(data, 'score')
        result = result.reset_index(drop=True)
        assert_frame_equal(expected, result)
