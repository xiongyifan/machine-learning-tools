"""class Pocket"""
from mlt.utils.logging import logger

from mlt.utils import weight
from mlt.utils import random
from mlt.stats import scores
from .pla import PLA


class Pocket(PLA):
    """Pocket model"""

    def __init__(self, order='sequence', correct_times=-1, learning_rate=1.0):
        super().__init__(order, correct_times, learning_rate)

    def fit(self, X, y, w=None, w_init_func=weight.init_zeros):
        """Fit Pocket model

        Parameters
        ----------
        X : ndarray
        y : ndarray
        w : ndarray
        w_init_func : ndarray

        Returns
        -------
        self
        """
        m, n = X.shape

        order = random.generate_sequence(self._order, m)

        self._init_weight(w, n, w_init_func)

        w_bast = weight.init_zeros(n)

        is_all_x_right = False
        correct_num = 0

        while is_all_x_right is not True:

            for i in order:
                x_one = X[i]

                y_one = y[i]

                y_predict = self._decision_function(x_one)

                if y_predict != y_one:
                    self._w += self._learning_rate * y_one * x_one
                    self._halt_step += 1
                    accuracy_bast = scores.cal_accuracy(X, y, w_bast)
                    accuracy = scores.cal_accuracy(X, y, self.w_)
                    if accuracy > accuracy_bast:
                        w_bast = self._w.copy()
                    correct_num = 0
                else:
                    correct_num += 1

                if correct_num == m or self.halt_step_ == self._correct_times:
                    is_all_x_right = True
                    break

        self._w = w_bast

        logger.info('-------------------------------------')
        logger.info('total _halt_step is %d', self.halt_step_)
        logger.info('the training accuracy is %s', scores.cal_accuracy(X, y, w_bast))
        logger.info('-------------------------------------')

        return self
