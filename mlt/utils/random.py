"""anything about random"""

import random


def generate_sequence(order, m):
    """generate a ﻿sequence

    Parameters
    ----------
    order : str
        one of list[sequence, random]
    m : int
        the max number
    Returns
    -------
    sequence : list
    """
    sequence = None
    if order is 'sequence':
        sequence = list(range(m))
    elif order is 'random':
        sequence = list(range(m))
        random.shuffle(sequence)

    return sequence
