"""class PLA"""

import numpy as np
import pandas as pd

from mlt.utils import weight
from mlt.stats import scores
from mlt.utils import random
from .model import Model


class PLA(Model):
    """PLA algorithm"""

    _halt_step = 0

    def __init__(self, order='sequence', correct_times=-1, learning_rate=1.0):
        self._order = order
        self._correct_times = correct_times
        self._learning_rate = learning_rate

    def fit(self, X, y, w):
        """Fit PLA model.

        Parameters
        ----------
        X : pd.DataFrame
        y : pd.DataFrame
        w : ndarray

        Returns
        -------
        self
        """
        m, n = X.shape

        order = random.generate_sequence(self._order, m)

        self._w = self._init_weight(w, weight.init_zeros, n)

        is_all_x_right = False
        correct_num = 0

        while is_all_x_right is not True:

            for i in order:
                x_one = X[i]
                y_one = y[i]
                y_predict = self._decision_function(x_one)

                if y_predict != y_one:
                    w += self._learning_rate * y_one * x_one
                    self._halt_step += 1
                    correct_num = 0
                    print('the training accuracy is ', scores.cal_accuracy(X, y, self.w_))
                else:
                    correct_num += 1

                if correct_num == m or self.halt_step_ == self._correct_times:
                    is_all_x_right = True
                    break

        print('-------------------------------------')
        print('total _halt_step is ', self.halt_step_)
        print('the training accuracy is ', scores.cal_accuracy(X, y, self.w_))
        print('-------------------------------------')

        return self

    def _decision_function(self, x):
        return np.sign(x.dot(self.w_))

    @property
    def halt_step_(self):
        """the step that the model stopped

        Returns
        -------
        int

        Notes
        -----
        getter for _halt_step
        """
        return self._halt_step

    @property
    def w_(self):
        """the weight

        Returns
        -------
        ndarray

        Notes
        -----
        getter for _w
        """
        return self._w
