"""test for pla.py"""

import unittest
import numpy as np
from mlt.models import PLA


# noinspection PyMethodMayBeStatic
class PLATestCase(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_PLA_fit(self):
        x_train, y_train, _, _ = self.__data_hw1_15()
        pla = PLA()
        pla.fit(x_train, y_train)
        expected = 45
        result = pla.halt_step_
        self.assertEqual(expected, result)

    def __data_hw1_15(self):
        x_y_train = np.fromfile('../data/hw1_15_train.dat', dtype=float, sep=' ').reshape(-1, 5)

        x_train = x_y_train[:, :4]

        x0 = np.ones(x_train.shape[0], dtype=float).reshape(-1, 1)

        x_train = np.hstack((x0, x_train))

        y_train = x_y_train[:, 4]

        m, n = x_train.shape

        return x_train, y_train, m, n