from sklearn.datasets import load_wine

from src.datasets.dataset_utils import load_and_save, load_from_sklearn


def main():
    load_data_fn = load_from_sklearn(load_wine)
    load_and_save('wine', load_data_fn)


if __name__ == '__main__':
    main()
