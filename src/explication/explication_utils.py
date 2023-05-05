from os.path import join, dirname
from time import time

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model

from src.datasets.dataset_utils import read_all_datasets
from src.models.model_utils import get_model_path
from src.solver.milp import build_network
from src.solver.tjeng import insert_tjeng_output_constraints


def _get_explication_path(*paths):
    return join(dirname(__file__), *paths)


def _ordinal(n: int):
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    else:
        suffix = ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]
    return str(n) + suffix


def _message_getting_explication(file, features, network_index, network_input, network_output):
    file.write(f'Getting explication for the {_ordinal(network_index + 1)} data:')
    for feature in features:
        file.write(f'\n- {feature} = {network_input[feature]}')
    file.write(f'\n\nWhy does the NN produce {network_output} as output? ...\n\n')


def _print_explication(file, data_index, explication, accumulated_time):
    file.write(f'Explication for the {_ordinal(data_index + 1)} data:')
    if explication['relevant'].size > 0:
        file.write(f'\n- Relevant: {explication["relevant"]}')
    if explication['irrelevant'].size > 0:
        file.write(f'\n- Irrelevant: {explication["irrelevant"]}')
    file.write(f'\n\nTime of explication: {explication["time"]:.2f} seconds')
    file.write(f'\nTotal time: {accumulated_time:.2f} seconds\n\n')


def _minimal_explication(mdl, bounds, network_input, network_output, features, accumulated_time):
    mdl = mdl.clone(new_name='clone')
    number_classes = len(bounds['output'])
    variables = {
        'output': [mdl.get_var_by_name(f'o_{index}') for index in range(number_classes)],
        'binary': mdl.binary_var_list(number_classes - 1, name='b')
    }
    input_constraints = mdl.add_constraints(
        [mdl.get_var_by_name(f'x_{index}') == feature for index, feature in enumerate(network_input)], names='input')
    mdl.add_constraint(mdl.sum(variables['binary']) >= 1)
    insert_tjeng_output_constraints(mdl, bounds['output'], network_output, variables)
    explication = {}
    explication_mask = np.ones_like(network_input, dtype=bool)
    start_time = time()
    for index, constraint in enumerate(input_constraints):
        mdl.remove_constraint(constraint)
        explication_mask[index] = False
        mdl.solve(log_output=False)
        if mdl.solution is not None:
            mdl.add_constraint(constraint)
            explication_mask[index] = True
    mdl.end()
    end_time = time()
    time_explication = end_time - start_time
    explication['relevant'] = np.array(features)[explication_mask]
    explication['irrelevant'] = np.array(features)[~explication_mask]
    explication['time'] = time_explication
    return explication, accumulated_time + time_explication


def get_minimal_explication(dataset_name, number_samples=None):
    with open(_get_explication_path(dataset_name, 'explication.txt'), 'w') as file:
        data = read_all_datasets(dataset_name)
        dataframe = pd.concat((data['train'], data['val'], data['test']), ignore_index=True)
        features = list(dataframe.columns)[:-1]
        model = load_model(get_model_path(dataset_name, f'{dataset_name}.h5'))
        layers = model.layers
        mdl, bounds = build_network(layers, dataframe)
        test = data['test'].head(number_samples) if number_samples else data['test']
        acc_time = 0
        for index, test_data in test.iterrows():
            network_input = test_data.iloc[:-1]
            network_output = np.argmax(model.predict(tf.reshape(network_input, (1, -1))))
            _message_getting_explication(file, features, index, network_input, network_output)
            explication, acc_time = _minimal_explication(mdl, bounds, network_input, network_output, features, acc_time)
            _print_explication(file, index, explication, acc_time)
        mdl.end()
