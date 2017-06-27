"""save data to file"""

import pandas as pd
import numpy as np


def to_csv(data, path):
    """make the csv form that is easy to upload to Kaggle

    Parameters
    ----------
    data : pd.DataFrame
    path : str
    """
    data.to_csv(path, index=False)


def train_val_test_to_npz(path, X_train, y_train, X_val, y_val, X_test, y_test):
    """when you use this function. please be sure all of your data type are numeral
    if there are strings in your data. I am not sure it is 100% correct function.
    because when I test the function. if the data contains strings. the assert function can not pass.

    Parameters
    ----------
    path :
    X_train : pd.DataFrame
    y_train : pd.DataFrame
    X_val : pd.DataFrame
    y_val : pd.DataFrame
    X_test : pd.DataFrame
    y_test : pd.DataFrame

    Returns
    -------

    """
    np.savez(path, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)
