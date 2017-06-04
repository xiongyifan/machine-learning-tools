import random


def get_order(order, m):
    if order is 'sequence':
        order = list(range(m))
    elif order is 'random':
        order = list(range(m))
        random.shuffle(order)

    return order
