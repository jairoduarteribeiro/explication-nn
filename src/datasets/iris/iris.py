from sklearn.datasets import load_iris

from src.datasets.dataset_utils import load_and_save


def main():
    load_and_save('iris', load_iris)


if __name__ == '__main__':
    main()
