"""initial weight"""

import numpy as np


def init_zeros(n):
    """initial a zero weight

    Parameters
    ----------
    n : the count of features

    Returns
    -------
    w : ndarray
    """
    w = np.zeros(n, dtype=float)
    return w


def init(func, n):
    """choose a func to init weight

    Parameters
    ----------
    func : str
        func_name
    n : int
        features or node number

    Returns
    -------
    w : ndarray
    """
    w = None
    if func is 'zeros':
        w = init_zeros(n)
    return w
