from os.path import join, dirname

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def get_dataset_path(*paths):
    return join(dirname(__file__), *paths)


def transform(x, x_columns=None):
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    return pd.DataFrame(x, columns=x_columns)


def split_dataset(x, y):
    x_train, x_test, y_train, y_test = \
        train_test_split(x, y, train_size=0.8, random_state=42, stratify=y)
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


def load_and_save(dataset_name, load_data_fn):
    x, y = load_data_fn(as_frame=True, return_X_y=True)
    x_columns = list(x.columns)
    x = transform(x, x_columns)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = split_dataset(x, y)
    train_csv_path = get_dataset_path(dataset_name, 'train.csv')
    validation_csv_path = get_dataset_path(dataset_name, 'validation.csv')
    test_csv_path = get_dataset_path(dataset_name, 'test.csv')
    save_dataset(x_train, y_train, train_csv_path)
    save_dataset(x_val, y_val, validation_csv_path)
    save_dataset(x_test, y_test, test_csv_path)
