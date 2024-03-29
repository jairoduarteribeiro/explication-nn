from os.path import join, dirname
from time import time

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model

from src.datasets.dataset_utils import read_all_datasets
from src.models.model_utils import get_model_path
from src.solver.box import box_relax_input_bounds, box_has_solution
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
    file.write(f'Getting explication for the {_ordinal(network_index + 1)} data:\n')
    for feature in features:
        file.write(f'- {feature} = {network_input[feature]}\n')
    file.write(f'\nWhy does the NN produce {network_output} as output? ...\n\n')


def _print_explication(file, data_index, explication):
    file.write(f'Explication for the {_ordinal(data_index + 1)} data:\n')
    if explication['relevant'].size > 0:
        file.write(f'- Relevant: {explication["relevant"]}\n')
    if explication['irrelevant'].size > 0:
        file.write(f'- Irrelevant: {explication["irrelevant"]}\n')
    if explication['calls_box'] > 0:
        file.write('\nBox results:\n')
        total = explication['calls_box'] + explication['calls_solver']
        percentage_box = 100 * explication['calls_box'] / total
        percentage_solver = 100 * explication['calls_solver'] / total
        file.write(f'- Calls to box: {explication["calls_box"]} ({percentage_box:.2f}%)\n')
        file.write(f'- Calls to solver: {explication["calls_solver"]} ({percentage_solver:.2f}%)\n')
        file.write(f'- Features irrelevant by box: {explication["solved_by_box"]}\n')
    file.write(f'\nTime of explication: {explication["time"]:.2f} seconds\n')
    file.write(f'--------------------------------------------------------------------------------\n')


def _print_final_metrics(file, acc_metrics):
    file.write(f'FINAL METRICS:\n')
    file.write(f'- Total time: {acc_metrics["total_time"]:.2f} seconds\n')
    if acc_metrics['total_calls_box'] > 0:
        total_calls = acc_metrics['total_calls_box'] + acc_metrics['total_calls_solver']
        percentage_box = 100 * acc_metrics['total_calls_box'] / total_calls
        percentage_solver = 100 * acc_metrics['total_calls_solver'] / total_calls
        file.write(f'- Total calls to box: {acc_metrics["total_calls_box"]} ({percentage_box:.2f})%\n')
        file.write(f'- Total calls to solver: {acc_metrics["total_calls_solver"]} ({percentage_solver:.2f})%\n')


def _minimal_explication(mdl, bounds, network_input, network_output, layers, features, acc_metrics, use_box):
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
    explication = {
        'calls_box': 0,
        'calls_solver': 0,
        'solved_by_box': []
    }
    explication_mask = np.ones_like(network_input, dtype=bool)
    start_time = time()
    for index, constraint in enumerate(input_constraints):
        mdl.remove_constraint(constraint)
        explication_mask[index] = False
        if use_box:
            relax_input_mask = ~explication_mask
            input_bounds = box_relax_input_bounds(network_input, bounds['input'], relax_input_mask)
            if box_has_solution(input_bounds, layers, network_output):
                explication['calls_box'] += 1
                explication['solved_by_box'].append(features[index])
                continue
        mdl.solve(log_output=False)
        explication['calls_solver'] += 1
        if mdl.solution is not None:
            mdl.add_constraint(constraint)
            explication_mask[index] = True
    mdl.end()
    end_time = time()
    time_explication = end_time - start_time
    explication['relevant'] = np.array(features)[explication_mask]
    explication['irrelevant'] = np.array(features)[~explication_mask]
    explication['time'] = time_explication
    acc_metrics['total_calls_box'] += explication['calls_box']
    acc_metrics['total_calls_solver'] += explication['calls_solver']
    acc_metrics['total_time'] += explication['time']
    return explication


def get_minimal_explication(dataset_name, use_box=True, number_samples=None, verbose=False):
    file_name = _get_explication_path(dataset_name, f'explication{"_with_box" if use_box else ""}.txt')
    with open(file_name, 'w') as file:
        data = read_all_datasets(dataset_name)
        dataframe = pd.concat((data['train'], data['val'], data['test']), ignore_index=True)
        features = list(dataframe.columns)[:-1]
        model = load_model(get_model_path(dataset_name, f'{dataset_name}.h5'))
        layers = model.layers
        mdl, bounds = build_network(layers, dataframe)
        test = data['test'].head(number_samples) if number_samples else data['test']
        acc_metrics = {
            'total_time': 0,
            'total_calls_box': 0,
            'total_calls_solver': 0
        }
        for index, test_data in test.iterrows():
            network_input = test_data.iloc[:-1]
            network_output = np.argmax(model.predict(tf.reshape(network_input, (1, -1))))
            if verbose:
                _message_getting_explication(file, features, index, network_input, network_output)
            explication = \
                _minimal_explication(mdl, bounds, network_input, network_output, layers, features, acc_metrics, use_box)
            if verbose:
                _print_explication(file, index, explication)
        _print_final_metrics(file, acc_metrics)
        mdl.end()
