import unittest
import pandas as pd
import numpy as np
import numpy.testing as npt
from pandas.util.testing import assert_frame_equal
import mlt

class NullValueTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_fill_null_value_with_median(self):
        data = pd.DataFrame(np.array([np.nan, 10.0, 11.0]), columns=['age'])
        expected = pd.DataFrame(np.array([10.5, 10.0, 11.0]), columns=['age'])
        result = mlt.preprocessing.fill_null_value_with_median(data, 'age')
        npt.assert_array_equal(expected, result)

    def test_remove_null_label_rows(self):
        data = pd.DataFrame(np.array([np.array([np.nan, 60.0]), np.array([10.0, np.nan]), np.array([11.0, 80.0])]),
                            columns=['age', 'score'])
        expected = pd.DataFrame(np.array([np.array([np.nan, 60.0]), np.array([11.0, 80.0])]), columns=['age', 'score'])
        result = mlt.preprocessing.remove_null_label_rows(data, 'score')
        result = result.reset_index(drop=True)
        print(expected.head(10))
        print(result.head(10))

        assert_frame_equal(expected, result)