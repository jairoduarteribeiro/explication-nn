from os.path import join, dirname

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def get_dataset_path(*paths):
    return join(dirname(__file__), *paths)


def _transform(x, x_columns=None):
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    return pd.DataFrame(x, columns=x_columns)


def _split_dataset(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42, stratify=y)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42, stratify=y_test)
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def _save_dataset(x, y, csv_path):
    csv = pd.concat((x, y), axis=1)
    csv.to_csv(csv_path, index=False)


def _read_dataset(csv_path, split=False):
    df = pd.read_csv(csv_path)
    return df if not split else (df.iloc[:, :-1], df.iloc[:, -1])


def read_all_datasets(dataset_name, split=False):
    train = _read_dataset(get_dataset_path(dataset_name, 'train.csv'), split)
    validation = _read_dataset(get_dataset_path(dataset_name, 'validation.csv'), split)
    test = _read_dataset(get_dataset_path(dataset_name, 'test.csv'), split)
    return {
        'name': dataset_name,
        'x_train': train[0],
        'y_train': train[1],
        'x_val': validation[0],
        'y_val': validation[1],
        'x_test': test[0],
        'y_test': test[1]
    } if split else {
        'name': dataset_name,
        'train': train,
        'val': validation,
        'test': test
    }


def load_from_sklearn(sklearn_load_fn):
    return lambda: sklearn_load_fn(return_X_y=True, as_frame=True)


def load_and_save(dataset_name, load_data_fn):
    x, y = load_data_fn()
    x_columns = list(x.columns)
    x = _transform(x, x_columns)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = _split_dataset(x, y)
    train_csv_path = get_dataset_path(dataset_name, 'train.csv')
    validation_csv_path = get_dataset_path(dataset_name, 'validation.csv')
    test_csv_path = get_dataset_path(dataset_name, 'test.csv')
    _save_dataset(x_train, y_train, train_csv_path)
    _save_dataset(x_val, y_val, validation_csv_path)
    _save_dataset(x_test, y_test, test_csv_path)
