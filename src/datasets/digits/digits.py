from sklearn.datasets import load_digits

from src.datasets.dataset_utils import load_and_save


def main():
    load_and_save('digits', load_digits)


if __name__ == '__main__':
    main()
