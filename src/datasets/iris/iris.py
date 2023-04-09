from sklearn.datasets import load_iris

from src.datasets.dataset_utils import transform, split_dataset, load_and_save


def load_data():
    x, y = load_iris(as_frame=True, return_X_y=True)
    x = transform(x, x.columns)
    return split_dataset(x, y)


def main():
    load_and_save('iris', load_data)


if __name__ == '__main__':
    main()
