"""save data to file"""

import pandas as pd


def to_csv(data, path):
    """make the csv form that is easy to upload to Kaggle

    Parameters
    ----------
    data : pd.DataFrame
    path : str
    """
    data.to_csv(path, index=False)