from src.models.model_utils import train


def main():
    nn_params = {'n_classes': 3, 'n_hidden_layers': 4, 'n_neurons': 16, 'n_epochs': 1000, 'batch_size': 4}
    train('iris', nn_params)


if __name__ == '__main__':
    main()
