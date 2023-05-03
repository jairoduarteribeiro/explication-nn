from src.explication.explication_utils import get_minimal_explication


def main():
    get_minimal_explication('digits', False, number_samples=20)


if __name__ == '__main__':
    main()
