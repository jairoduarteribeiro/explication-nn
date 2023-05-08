from sklearn.datasets import load_breast_cancer

from src.datasets.dataset_utils import load_and_save, load_from_sklearn


def main():
    load_data_fn = load_from_sklearn(load_breast_cancer)
    load_and_save('breast_cancer', load_data_fn)


if __name__ == '__main__':
    main()
