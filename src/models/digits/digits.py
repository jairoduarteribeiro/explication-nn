from src.models.model_utils import train


def main():
    nn_params = {
        'n_classes': 10, 'n_hidden_layers': 8, 'n_neurons': 32, 'n_epochs': 10000, 'batch_size': 8, 'patience': 100
    }
    train('digits', nn_params)


if __name__ == '__main__':
    main()
