import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import load_model
from src.solver.milp import build_network
from src.models.model_utils import get_model_path
from src.datasets.dataset_utils import get_dataset_path, read_dataset
from src.explication.explication_utils import get_minimal_explication, print_explication


def main():
    train_data = read_dataset(get_dataset_path('iris', 'train.csv'))
    validation_data = read_dataset(get_dataset_path('iris', 'validation.csv'))
    test_data = read_dataset(get_dataset_path('iris', 'test.csv'))
    features = test_data.columns[: -1]
    dataframe = pd.concat([train_data, validation_data, test_data], ignore_index=True)
    model = load_model(get_model_path('iris', 'iris.h5'))
    layers = model.layers
    mdl, bounds = build_network(layers, dataframe, 'fischetti')
    for data_index, data in test_data.iterrows():
        print(f'Getting explication for data {data_index}...')
        network_input = tf.reshape(data.iloc[:-1], (1, -1))
        network_output = np.argmax(model.predict(network_input))
        explanation = get_minimal_explication(mdl, bounds, 'fischetti', network_input, network_output)
        print_explication(data_index, features, explanation)


if __name__ == '__main__':
    main()
