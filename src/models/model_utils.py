from keras.models import Sequential
from keras.layers import InputLayer, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from os.path import join, dirname
from operator import itemgetter
from time import time
from src.datasets.dataset_utils import get_dataset_path, read_dataset


def get_model_path(*paths):
    return join(dirname(__file__), *paths)


def train(dataset_name, nn_params):
    x_train, y_train = read_dataset(get_dataset_path(dataset_name, 'train.csv'), split=True)
    x_val, y_val = read_dataset(get_dataset_path(dataset_name, 'validation.csv'), split=True)
    n_classes, n_hidden_layers, n_neurons, n_epochs, batch_size = \
        itemgetter('n_classes', 'n_hidden_layers', 'n_neurons', 'n_epochs', 'batch_size')(nn_params)
    model = Sequential(layers=(InputLayer(input_shape=(x_train.shape[1],)),
                               *(Dense(n_neurons, activation='relu') for _ in range(n_hidden_layers - 1)),
                               Dense(n_classes, activation='softmax')))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
    callbacks = (EarlyStopping(patience=int(n_epochs * 0.05)),
                 ModelCheckpoint(filepath=get_model_path(dataset_name, f'{dataset_name}.h5'), save_best_only=True))
    start_time = time()
    model.fit(x_train, y_train, epochs=n_epochs, batch_size=batch_size, validation_data=(x_val, y_val),
              callbacks=callbacks)
    end_time = time()
    print(f'Time of training: {end_time - start_time:.2f} seconds.')
