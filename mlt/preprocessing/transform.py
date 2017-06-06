"""transform features"""

import pandas as pd

from mlt.utils.logging import logger


def one_hot_encoder(X, categorical_features):
    """Convert categorical feature into dummy/indicator features

    Parameters
    ----------
    X : pd.DataFrame
    categorical_features : list

    Returns
    -------
    X : pd.DataFrame
    """
    _, n = X.shape
    for label_name in categorical_features:
        dummies = pd.get_dummies(X[label_name], prefix=label_name, dummy_na=False)
        X = X.drop(label_name, 1)
        X = pd.concat([X, dummies], axis=1)
    logger.info('before %d features, after one_hot_encoder %d features', n, X.shape[1])
    return X
