"""read data from file"""

import pandas as pd
import numpy as np


def from_csv(path, na_values=None):
    """read csv from a path

    Parameters
    ----------
    path : str
    na_values : list[str]

    Returns
    -------

    """
    return pd.read_csv(path, na_values=na_values)


def from_npz_train_val_test_x_y(path):
    """read npz and split to train, val, test

    Parameters
    ----------
    path : str

    Returns
    -------

    """
    data = np.load(path)
    return data['X_train'], data['y_train'], data['X_val'], data['y_val'], data['X_test'], data['y_test']
