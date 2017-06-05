"""split data in different way"""

from sklearn.model_selection import train_test_split
import pandas as pd


def split_data(data, percentage):
    # todo: move to model_selection
    """split data into 2 parts

    Parameters
    ----------
    data : pd.DataFrame
    percentage : float

    Returns
    -------
    pd.DataFrame
    """
    return train_test_split(data, test_size=percentage)
