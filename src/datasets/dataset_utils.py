import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from os.path import join, dirname


def get_dataset_path(*paths):
    return join(dirname(__file__), *paths)


def transform(x, x_columns=None):
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    return pd.DataFrame(x, columns=x_columns)


def split_dataset(x, y, with_validation=False):
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42, stratify=y)
    if not with_validation:
        return (x_train, y_train), (x_test, y_test)
    x_val, x_test, y_val, y_test = \
        train_test_split(x_test, y_test, test_size=0.5, random_state=42, stratify=y_test)
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def save_dataset(x, y, csv_path):
    csv = pd.concat([x, y], axis=1)
    csv.to_csv(csv_path, index=False)


def read_dataset(csv_path, split=False):
    df = pd.read_csv(csv_path)
    if not split:
        return df
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return x, y
