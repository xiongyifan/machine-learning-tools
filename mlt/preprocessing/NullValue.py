import pandas as pd


def remove_null_label_rows(data, label_name) -> pd.DataFrame:
    count_null_label = data[label_name].isnull().sum()
    if count_null_label != 0:
        data = data[data[label_name].notnull()]
    reason = str(count_null_label) + ' null label rows are removed'
    print(reason)
    return data


def fill_null_value_with_median(data, label_name) -> pd.DataFrame:
    count_null_value = data[label_name].isnull().sum()
    fill_value = data[label_name].median()
    temp = data[label_name].fillna(fill_value)
    data = data.drop(label_name, 1)
    data = pd.concat([data, temp], axis=1)
    reason = str(count_null_value) + ' null values are filled in the column ' + label_name
    print(reason)
    return data


def fill_null_value_with_highest_probability_item(data, label_name):
    count_null_value = data[label_name].isnull().sum()
    fill_value = data[label_name].value_counts().index[0]
    temp = data[label_name].fillna(fill_value)
    data = data.drop(label_name, 1)
    data = pd.concat([data, temp], axis=1)
    reason = str(count_null_value) + ' null values are filled in the column ' + label_name
    print(reason)
    return data