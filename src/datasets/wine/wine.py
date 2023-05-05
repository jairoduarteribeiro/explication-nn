from sklearn.datasets import load_wine

from src.datasets.dataset_utils import load_and_save


def main():
    load_and_save('wine', load_wine)


if __name__ == '__main__':
    main()
