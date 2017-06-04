from .Model import Model
import utils
import numpy as np
import mlt

class PLA(Model):

    halt_step = 0

    def __init__(self, w=None, order='sequence', correct_times=-1, learning_rate=1.0):
        self.order = order
        self.correct_times = correct_times
        self.learning_rate = learning_rate
        self.w = w

    def fit(self, X, Y):
        m, n = X.shape

        order = utils.get_order(self.order, m)

        if self.w is None:
            self.w = utils.init_w(n)

        is_all_x_right = False
        correct_num = 0

        while is_all_x_right is not True:

            for i in order:
                x = X[i]
                y = Y[i]
                y_pred = np.sign(x.dot(self.w))

                if y_pred != y:
                    self.w = self.w + self.learning_rate * y * x
                    self.halt_step += 1
                    correct_num = 0
                    print('the training accuracy is ', mlt.stats.cal_accuracy(X, Y, self.w))
                else:
                    correct_num += 1

                if correct_num == m or self.halt_step == self.correct_times:
                    is_all_x_right = True
                    break

        print('-------------------------------------')
        print('total halt_step is ', self.halt_step)
        print('the training accuracy is ', mlt.stats.cal_accuracy(X, Y, self.w))
        print('-------------------------------------')

        return self
