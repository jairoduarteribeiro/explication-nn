from sklearn.datasets import load_iris

from src.datasets.dataset_utils import load_and_save, load_from_sklearn


def main():
    load_data_fn = load_from_sklearn(load_iris)
    load_and_save('iris', load_data_fn)


if __name__ == '__main__':
    main()
