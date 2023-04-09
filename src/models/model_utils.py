from keras.models import Sequential, Model
from keras.layers import InputLayer, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from os.path import join, dirname
from operator import itemgetter
from time import time
from dataclasses import dataclass
from src.datasets.dataset_utils import get_dataset_path, read_dataset


def get_model_path(*paths):
    return join(dirname(__file__), *paths)


@dataclass
class NNParams:
    classes: int
    hidden_layers: int
    neurons: int
    epochs: int
    batch_size: int


def train(dataset_name, nn_params: NNParams):
    x_train, y_train = read_dataset(get_dataset_path(dataset_name, 'train.csv'), split=True)
    x_val, y_val = read_dataset(get_dataset_path(dataset_name, 'validation.csv'), split=True)
    model = Sequential()
    model.add(Dense(nn_params.neurons, input_shape=(x_train.shape[1],), activation='relu'))
    for _ in range(1, nn_params.hidden_layers - 1):
        model.add(Dense(nn_params.neurons, activation='relu'))
    model.add(Dense(nn_params.classes, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
    callbacks = (EarlyStopping(patience=int(nn_params.epochs * 0.05)),
                 ModelCheckpoint(filepath=get_model_path(dataset_name, f'{dataset_name}.h5'), save_best_only=True))
    start_time = time()
    model.fit(x_train, y_train, epochs=nn_params.epochs, batch_size=nn_params.batch_size, verbose=0,
              validation_data=(x_val, y_val), callbacks=callbacks)
    end_time = time()
    print(f'Time of training: {end_time - start_time:.2f} seconds.')
