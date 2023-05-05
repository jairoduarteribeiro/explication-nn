from dataclasses import dataclass
from os.path import join, dirname
from time import time

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense
from keras.models import Sequential

from src.datasets.dataset_utils import read_all_datasets


def get_model_path(*paths):
    return join(dirname(__file__), *paths)


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
    model.add(Dense(nn_params.neurons, input_shape=input_shape, activation='relu'))
    for _ in range(1, nn_params.hidden_layers - 1):
        model.add(Dense(nn_params.neurons, activation='relu'))
    model.add(Dense(nn_params.classes, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=('sparse_categorical_accuracy',))
    return model


def _train_and_save(data, nn_params: NNParams, model):
    callbacks = (EarlyStopping(patience=int(nn_params.epochs * 0.05)),
                 ModelCheckpoint(filepath=get_model_path(data['name'], f'{data["name"]}.h5'), save_best_only=True))
    start_time = time()
    model.fit(data['x_train'], data['y_train'], epochs=nn_params.epochs, batch_size=nn_params.batch_size, verbose=0,
              validation_data=(data['x_val'], data['y_val']), callbacks=callbacks)
    end_time = time()
    print(f'Time of training: {end_time - start_time:.2f} seconds.')
    print('Evaluate on test data:')
    results = model.evaluate(data['x_test'], data['y_test'], batch_size=nn_params.batch_size, verbose=0)
    print(f'- Loss: {results[0]}')
    print(f'- Accuracy: {results[1] * 100.0:.2f}%')


def train(dataset_name, nn_params: NNParams):
    data = read_all_datasets(dataset_name, split=True)
    model = _create_model(data, nn_params)
    _train_and_save(data, nn_params, model)
