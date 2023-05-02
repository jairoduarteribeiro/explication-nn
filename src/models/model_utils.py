from dataclasses import dataclass
from os.path import join, dirname
from time import time

from keras.activations import relu, softmax
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy
from keras.models import Sequential
from keras.optimizers import Adam

from src.datasets.dataset_utils import get_dataset_path, read_dataset


def get_model_path(*paths):
    return join(dirname(__file__), *paths)


def _read_datasets(dataset_name):
    x_train, y_train = read_dataset(get_dataset_path(dataset_name, 'train.csv'), split=True)
    x_val, y_val = read_dataset(get_dataset_path(dataset_name, 'validation.csv'), split=True)
    return {
        'name': dataset_name,
        'x_train': x_train,
        'y_train': y_train,
        'x_val': x_val,
        'y_val': y_val
    }


@dataclass
class NNParams:
    classes: int
    hidden_layers: int
    neurons: int
    epochs: int
    batch_size: int


def _create_model(data, nn_params: NNParams):
    input_shape = (data['x_train'].shape[1],)
    model = Sequential()
    model.add(Dense(nn_params.neurons, input_shape=input_shape, activation=relu))
    for _ in range(1, nn_params.hidden_layers - 1):
        model.add(Dense(nn_params.neurons, activation=relu))
    model.add(Dense(nn_params.classes, activation=softmax))
    model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(), metrics=[SparseCategoricalAccuracy()])
    return model


def _train_and_save(data, nn_params: NNParams, model):
    callbacks = (EarlyStopping(patience=int(nn_params.epochs * 0.05)),
                 ModelCheckpoint(filepath=get_model_path(data['name'], f'{data["name"]}.h5'), save_best_only=True))
    start_time = time()
    model.fit(data['x_train'], data['y_train'], epochs=nn_params.epochs, batch_size=nn_params.batch_size, verbose=0,
              validation_data=(data['x_val'], data['y_val']), callbacks=callbacks)
    end_time = time()
    print(f'Time of training: {end_time - start_time:.2f} seconds.')


def train(dataset_name, nn_params: NNParams):
    data = _read_datasets(dataset_name)
    model = _create_model(data, nn_params)
    _train_and_save(data, nn_params, model)
