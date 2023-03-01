from time import time
from keras.models import Sequential
from keras.layers import InputLayer, Dense
from keras.activations import relu, softmax
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy
from keras.callbacks import EarlyStopping, ModelCheckpoint
from os.path import join, dirname
from src.datasets.dataset_utils import get_dataset_path, read_dataset


def get_model_path(*paths):
    return join(dirname(__file__), *paths)


def train(dataset_name, nn_params):
    x_train, y_train = read_dataset(get_dataset_path(dataset_name, 'train.csv'), split=True)
    x_val, y_val = read_dataset(get_dataset_path(dataset_name, 'validation.csv'), split=True)
    model = Sequential([InputLayer(input_shape=(x_train.shape[1],))])
    for _ in range(nn_params['n_hidden_layers'] - 1):
        model.add(Dense(nn_params['n_neurons'], activation=relu))
    model.add(Dense(nn_params['n_classes'], activation=softmax))
    model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(), metrics=[SparseCategoricalAccuracy()])
    early_stopping = EarlyStopping(patience=nn_params['patience'])
    model_checkpoint = ModelCheckpoint(filepath=get_model_path(dataset_name, f'{dataset_name}.h5'), save_best_only=True)
    start_time = time()
    model.fit(x_train, y_train, epochs=nn_params['n_epochs'], batch_size=nn_params['batch_size'],
              validation_data=(x_val, y_val), callbacks=[early_stopping, model_checkpoint])
    end_time = time()
    print(f'Time of training: {end_time - start_time:.2f} seconds.')
