from sklearn.datasets import load_digits
from src.datasets.dataset_utils import get_dataset_path, transform, split_dataset, save_dataset


def load_data(split_with_validation=False):
    x, y = load_digits(as_frame=True, return_X_y=True)
    x = transform(x, x.columns)
    return split_dataset(x, y, split_with_validation)


def main():
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data(split_with_validation=True)
    train_csv_path = get_dataset_path('digits', 'train.csv')
    validation_csv_path = get_dataset_path('digits', 'validation.csv')
    test_csv_path = get_dataset_path('digits', 'test.csv')
    save_dataset(x_train, y_train, train_csv_path)
    save_dataset(x_val, y_val, validation_csv_path)
    save_dataset(x_test, y_test, test_csv_path)


if __name__ == '__main__':
    main()
