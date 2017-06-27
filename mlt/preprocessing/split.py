"""split data in different way"""

from sklearn.model_selection import train_test_split
import pandas as pd


def split_data(*arrays, percentage):
    # todo: move to model_selection
    """split data into 2 parts

    Parameters
    ----------
    arrays : sequence of indexables with same length / shape[0]
        Allowed inputs are lists, numpy arrays, scipy-sparse
        matrices or pandas dataframes.
    percentage : float

    Returns
    -------
    pd.DataFrame
    """
    return train_test_split(*arrays, test_size=percentage)


def split_train_val_test(*arrays, val_percentage, test_percentage):
    """split data to train, val, test

    Parameters
    ----------
    arrays :
    val_percentage :
    test_percentage :

    Returns
    -------
    X_train : pd.DataFrame
    """
    m, _ = arrays[0].shape
    val_size = int(m * val_percentage)
    test_size = int(m * test_percentage)
    train_size = int(m - val_size - test_size)

    if len(arrays) == 1:
        train, test = train_test_split(*arrays, train_size=train_size + val_size, test_size=test_size)
        train, val = train_test_split(train, train_size=train_size, test_size=val_size)
        result = (train, val, test)
    else:
        X_train, X_test, y_train, y_test = train_test_split(*arrays, train_size=train_size + val_size,
                                                            test_size=test_size)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=train_size,
                                                          test_size=val_size)
        result = (X_train, y_train, X_val, y_val, X_test, y_test)

    return result


def separate_x_y(data, label_name):
    """separate x and y from a pd.DataFrame

    Parameters
    ----------
    data : pd.DataFrame
    label_name : str

    Returns
    -------

    """
    y = data[label_name]
    X = data.drop(labels=[label_name], axis=1)
    return X, y
