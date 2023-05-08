from sklearn.datasets import load_digits

from src.datasets.dataset_utils import load_and_save, load_from_sklearn


def main():
    load_data_fn = load_from_sklearn(load_digits)
    load_and_save('digits', load_data_fn)


if __name__ == '__main__':
    main()
