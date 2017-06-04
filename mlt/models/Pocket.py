import numpy as np
import utils


def Pocket(x_train, y_train, order=None, correct_times=-1, learning_rate=1.0, x_test=None, y_test=None):
    m, n = x_train.shape

    order = utils.get_order(order, m)

    w = utils.init_w(n)

    w_bast = utils.init_w(n)

    is_all_x_right = False
    halt_step = 0
    correct_num = 0

    while is_all_x_right is not True:

        for i in order:
            x = x_train[i]

            y = y_train[i]

            y_pred = np.sign(x.dot(w))

            if y_pred != y:
                w = w + learning_rate * y * x
                halt_step += 1
                accuracy_bast = utils.cal_accuracy(x_train, y_train, w_bast)
                accuracy = utils.cal_accuracy(x_train, y_train, w)
                if accuracy > accuracy_bast:
                    w_bast = w
                correct_num = 0

                utils.verification(x_train, y_train, x_test, y_test, w_bast)

            else:
                correct_num += 1

            if correct_num == m or halt_step == correct_times:
                is_all_x_right = True
                break

    print('-------------------------------------')
    print('total halt_step is ', halt_step)
    print('the training accuracy is ', utils.cal_accuracy(x_train, y_train, w_bast))
    print('-------------------------------------')

    return w_bast, halt_step
