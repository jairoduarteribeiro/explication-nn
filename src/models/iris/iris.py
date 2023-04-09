from src.models.model_utils import train, NNParams


def main():
    nn_params = NNParams(classes=3, hidden_layers=4, neurons=16, epochs=1000, batch_size=4)
    train('iris', nn_params)


if __name__ == '__main__':
    main()
