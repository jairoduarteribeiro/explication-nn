import pandas as pd
from src.datasets.dataset_utils import get_dataset_path, transform, split_dataset, save_dataset


def load_data(split_with_validation=False):
    path = get_dataset_path('wine_quality_red', 'wine-quality-red.csv')
    column_names = ('fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides',
                    'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality')
    df = pd.read_csv(path, header=0, names=column_names, sep=';')
    x = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]
    x = transform(x, x.columns)
    return split_dataset(x, y, split_with_validation)


def main():
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data(split_with_validation=True)
    train_csv_path = get_dataset_path('wine_quality_red', 'train.csv')
    validation_csv_path = get_dataset_path('wine_quality_red', 'validation.csv')
    test_csv_path = get_dataset_path('wine_quality_red', 'test.csv')
    save_dataset(x_train, y_train, train_csv_path)
    save_dataset(x_val, y_val, validation_csv_path)
    save_dataset(x_test, y_test, test_csv_path)


if __name__ == '__main__':
    main()
