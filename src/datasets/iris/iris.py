import pandas as pd
from src.datasets.dataset_utils import get_dataset_path, transform, split_dataset, save_dataset


def load_data(split_with_validation=False):
    path = get_dataset_path('iris', 'iris.csv')
    column_names = ('id', 'sepal_length_cm', 'sepal_width_cm', 'petal_length_cm', 'petal_width_cm', 'target')
    df = pd.read_csv(path, header=0, names=column_names)
    df.loc[df.target == 'Iris-setosa', 'target'] = 0
    df.loc[df.target == 'Iris-versicolor', 'target'] = 1
    df.loc[df.target == 'Iris-virginica', 'target'] = 2
    df['target'] = df['target'].astype('int')
    x = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]
    x = transform(x, x.columns)
    return split_dataset(x, y, split_with_validation)


if __name__ == '__main__':
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data(split_with_validation=True)
    train_csv_path = get_dataset_path('iris', 'train.csv')
    validation_csv_path = get_dataset_path('iris', 'validation.csv')
    test_csv_path = get_dataset_path('iris', 'test.csv')
    save_dataset(x_train, y_train, train_csv_path)
    save_dataset(x_val, y_val, validation_csv_path)
    save_dataset(x_test, y_test, test_csv_path)
