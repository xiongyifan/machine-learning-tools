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
