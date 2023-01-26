from time import time
from keras.models import Sequential
from keras.layers import InputLayer, Dense
from keras.activations import relu, softmax
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy
from keras.callbacks import EarlyStopping, ModelCheckpoint
from src.datasets.dataset_utils import get_dataset_path, read_dataset
from src.models.model_utils import get_model_path


def main():
    x_train, y_train = read_dataset(get_dataset_path('iris', 'train.csv'))
    x_val, y_val = read_dataset(get_dataset_path('iris', 'validation.csv'))
    n_classes = 3
    n_hidden_layers = 4
    n_neurons = 16
    n_epochs = 1000
    batch_size = 4
    model = Sequential([InputLayer(input_shape=(x_train.shape[1],))])
    for _ in range(n_hidden_layers - 1):
        model.add(Dense(n_neurons, activation=relu))
    model.add(Dense(n_classes, activation=softmax))
    model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(), metrics=[SparseCategoricalAccuracy()])
    early_stopping = EarlyStopping(patience=20)
    model_checkpoint = ModelCheckpoint(filepath=get_model_path('iris', 'iris.h5'), save_best_only=True)
    start_time = time()
    model.fit(x_train, y_train, epochs=n_epochs, batch_size=batch_size,
              validation_data=(x_val, y_val), callbacks=[early_stopping, model_checkpoint])
    end_time = time()
    print(f'Time of training: {end_time - start_time:.2f} seconds.')


if __name__ == '__main__':
    main()
