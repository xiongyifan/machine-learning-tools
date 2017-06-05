"""deal with null value"""

from mlt.utils.logging import logger
import pandas as pd


def remove_rows_by_column(data, column_name):
    """remove the rows they have any null value

    Parameters
    ----------
    data : pd.DataFrame
    column_name : str

    Returns
    -------
    data : pd.DataFrame
    """
    count_null_label = data[column_name].isnull().sum()
    if count_null_label != 0:
        data = data[data[column_name].notnull()]
    reason = str(count_null_label) + ' null label rows are removed'
    logger.info(reason)
    return data


def fill_median(data, column_name):
    """fill null value with median

    Parameters
    ----------
    data : pd.DataFrame
    column_name : str

    Returns
    -------
    data : pd.DataFrame
    """
    count_null_value = data[column_name].isnull().sum()
    fill_value = data[column_name].median()
    temp = data[column_name].fillna(fill_value)
    data = data.drop(column_name, 1)
    data = pd.concat([data, temp], axis=1)
    reason = str(count_null_value) + ' null values are filled in the column ' + column_name
    logger.info(reason)
    return data


def fill_highest_probability_item(data, label_name):
    """fill null value with highest probability item

    Parameters
    ----------
    data : pd.DataFrame
    label_name : str

    Returns
    -------
    data : pd.DataFrame
    """
    count_null_value = data[label_name].isnull().sum()
    fill_value = data[label_name].value_counts().index[0]
    temp = data[label_name].fillna(fill_value)
    data = data.drop(label_name, 1)
    data = pd.concat([data, temp], axis=1)
    reason = str(count_null_value) + ' null values are filled in the column ' + label_name
    logger.info(reason)
    return data
