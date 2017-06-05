"""summary data"""

import pandas as pd
import numpy as np

from mlt.utils.logging import logger


def summary_data(data):
    """summary the data

    Parameters
    ----------
    data : pd.DataFrame
    """
    m, n = data.shape
    null_count = data.isnull().sum()
    null_percentage = null_count / pd.Series(np.ones([n]) * m, index=null_count.index)
    null_percentage = null_percentage.apply(lambda x: "{:.2%}".format(x))
    logger.info('shape:', data.shape)
    logger.info('--------------------------------------')
    d = {'type': data.dtypes, 'null_count': null_count, 'null_percentage': null_percentage}
    df = pd.DataFrame(d)
    logger.info(df)
