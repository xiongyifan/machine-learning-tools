import pandas as pd
import numpy as np


def summary_data(data: pd.DataFrame):
    m, n = data.shape
    null_count = data.isnull().sum()
    null_percentage = null_count / pd.Series(np.ones([n]) * m, index=null_count.index)
    null_percentage = null_percentage.apply(lambda x: "{:.2%}".format(x))
    print('shape:', data.shape)
    print('--------------------------------------')
    d = {'type': data.dtypes, 'null_count': null_count, 'null_percentage': null_percentage}
    df = pd.DataFrame(d)
    print(df)
