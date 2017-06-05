"""class PLA"""

import numpy as np
import pandas as pd

from mlt.utils import weight
from mlt.stats import scores
from mlt.utils import random
from .model import Model


class PLA(Model):
    """PLA algorithm"""

    halt_step = 0

    def __init__(self, w=None, order='sequence', correct_times=-1, learning_rate=1.0):
        self.order = order
        self.correct_times = correct_times
        self.learning_rate = learning_rate
        self.w = w  # todo: w should not be a variable. it should be a parameter

    def fit(self, X, y):
        """Fit PLA model.

        Parameters
        ----------
        X : pd.DataFrame
            Training data
        y : pd.DataFrame
            Target values

        Returns
        -------

        """
        m, n = X.shape

        order = random.generate_sequence(self.order, m)

        if self.w is None:
            self.w = weight.init_zeros(n)

        is_all_x_right = False
        correct_num = 0

        while is_all_x_right is not True:

            for i in order:
                x_one = X[i]
                y_one = y[i]
                y_predict = np.sign(x_one.dot(self.w))

                if y_predict != y_one:
                    self.w = self.w + self.learning_rate * y_one * x_one
                    self.halt_step += 1
                    correct_num = 0
                    print('the training accuracy is ', scores.cal_accuracy(X, y, self.w))
                else:
                    correct_num += 1

                if correct_num == m or self.halt_step == self.correct_times:
                    is_all_x_right = True
                    break

        print('-------------------------------------')
        print('total halt_step is ', self.halt_step)
        print('the training accuracy is ', scores.cal_accuracy(X, y, self.w))
        print('-------------------------------------')

        return self
