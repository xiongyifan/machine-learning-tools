"""transform features"""

import pandas as pd


def one_hot_encoder(X, categorical_features):
    """Convert categorical feature into dummy/indicator features

    Parameters
    ----------
    X : pd.DataFrame
    categorical_features : str

    Returns
    -------
    X : pd.DataFrame
    """
    for label_name in categorical_features:
        dummies = pd.get_dummies(X[label_name], prefix=label_name, dummy_na=False)
        X = X.drop(label_name, 1)
        X = pd.concat([X, dummies], axis=1)
    return X
