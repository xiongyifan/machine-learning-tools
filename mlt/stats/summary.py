"""summary data"""

import pandas as pd
import numpy as np

from mlt.utils.logging import logger


def __null(data):
    m, n = data.shape
    null_count = data.isnull().sum()
    null_percentage = null_count / m
    null_percentage = null_percentage.apply(lambda x: "{:.2%}".format(x))
    return {'null_count': null_count, 'null_percentage': null_percentage}


def __recommend(summary):
    null_fill_median = []
    null_fill_highest_probability_item = []
    null_no_advise = []

    for i in summary.index:
        t = str(summary['type'][i])
        if t == 'object':
            null_fill_highest_probability_item.append(i)
        elif t.startswith('float'):
            null_fill_median.append(i)
        else:
            null_no_advise.append(i)

    logger.info('!!!recommend!!! : call preprocessing.null_value.column_fill_median() at column %s', null_fill_median)
    logger.info('!!!recommend!!! : call preprocessing.null_value.column_fill_highest_probability_item() at column %s',
                null_fill_highest_probability_item)
    logger.info('!!!recommend!!! : no advise at column %s, they are int, but maybe someone should belong to category',
                null_no_advise)


def __type(data):
    return {'type': [t.name for t in data.dtypes]}


def __generate_summary(dicts, sort_by, ascending):
    d_concat = {}
    for d in dicts:
        d_concat.update(d)

    return pd.DataFrame(d_concat).sort_values(by=sort_by, ascending=ascending)


def __category(data):
    """

    Parameters
    ----------
    data : pd.DataFrame

    Returns
    -------

    """
    m, _ = data.shape
    count_categories = []
    for col_name in data.columns:
        # if not data[col_name].dtypes.name.startswith('float'):
        unique_cat = len(data[col_name].unique())
        count_categories.append(unique_cat)

        # next 3 lines can cal std for category
        # std_category = (data[col_name].value_counts() / m).std()
        # std_categories.append(std_category)
    # return {'count_category': count_categories, 'std_category': std_categories}
    return {'count_category': count_categories}


def summary_data(data, sort_by='null_count', ascending=False):
    """summary the data

    Parameters
    ----------
    data : pd.DataFrame
    sort_by : str
    ascending : bool
    """
    # todo: positive and native percentage
    # todo: category std

    null_dict = __null(data)
    type_dict = __type(data)
    category_dict = __category(data)

    summary = __generate_summary([null_dict, category_dict, type_dict], sort_by, ascending)

    print(summary.head(100000))

    __recommend(summary)
