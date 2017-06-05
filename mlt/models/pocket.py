"""class Pocket"""

from mlt.utils import weight
from mlt.utils import random
from mlt.stats import scores
from .pla import PLA


class Pocket(PLA):
    """Pocket model"""

    def __init__(self, order='sequence', correct_times=-1, learning_rate=1.0):
        super().__init__(order, correct_times, learning_rate)

    def fit(self, X, y, w):
        """Fit Pocket model

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

        order = random.generate_sequence(self.order, m)

        self.w = self._init_weight(w, weight.init_zeros, n)

        w_bast = weight.init_zeros(n)

        is_all_x_right = False
        correct_num = 0

        while is_all_x_right is not True:

            for i in order:
                x_one = X[i]

                y_one = y[i]

                y_predict = self._decision_function(x_one)

                if y_predict != y_one:
                    w += self.learning_rate * y_one * x_one
                    self._halt_step += 1
                    accuracy_bast = scores.cal_accuracy(X, y, w_bast)
                    accuracy = scores.cal_accuracy(X, y, w)
                    if accuracy > accuracy_bast:
                        w_bast = w
                    correct_num = 0
                else:
                    correct_num += 1

                if correct_num == m or self.halt_step == self.correct_times:
                    is_all_x_right = True
                    break

        print('-------------------------------------')
        print('total _halt_step is ', self.halt_step)
        print('the training accuracy is ', scores.cal_accuracy(X, y, w_bast))
        print('-------------------------------------')

        return self
