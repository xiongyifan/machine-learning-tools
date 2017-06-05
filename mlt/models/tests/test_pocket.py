"""test for pocket.py"""

import unittest
import numpy as np
from mlt.models import Pocket
from mlt.stats import scores
from mlt.utils.logging import logger


# noinspection PyMethodMayBeStatic
class PocketTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_PLA_fit(self):
        x_train, y_train, x_test, y_test, m, n = self.__data_hw1_18()

        times = 2000
        error_rates = np.zeros(times)
        for i in range(times):
            pocket = Pocket('random', 50, 1)
            pocket.fit(x_train, y_train)
            error_rates[i] = scores.cal_error_rate(x_test, y_test, pocket.w_)

        result = np.average(error_rates)
        logger.info('error rate is %f', result)
        # noinspection PyTypeChecker
        self.assertTrue(0.12 < result < 0.14)

    def __data_hw1_18(self):
        x_y_train = np.fromfile('../data/hw1_18_train.dat', dtype=float, sep=' ').reshape(-1, 5)
        x_y_test = np.fromfile('../data/hw1_18_test.dat', dtype=float, sep=' ').reshape(-1, 5)

        x_train = x_y_train[:, :4]
        x_test = x_y_test[:, :4]

        x0 = np.ones(x_train.shape[0], dtype=float).reshape(-1, 1)

        x_train = np.hstack((x0, x_train))
        x_test = np.hstack((x0, x_test))

        y_train = x_y_train[:, 4]
        y_test = x_y_test[:, 4]

        m, n = x_train.shape

        return x_train, y_train, x_test, y_test, m, n
