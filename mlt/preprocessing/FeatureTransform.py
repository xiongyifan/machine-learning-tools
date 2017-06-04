import pandas as pd


def OneHotEncoder(X, categorical_features):
    for label_name in categorical_features:
        dummies = pd.get_dummies(X[label_name], prefix=label_name, dummy_na=False)
        X = X.drop(label_name, 1)
        X = pd.concat([X, dummies], axis=1)
    return X
