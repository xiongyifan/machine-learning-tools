from sklearn.model_selection import train_test_split


def split_data(data, percentage):
    return train_test_split(data, test_size=percentage)
