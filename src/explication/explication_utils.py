import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from src.datasets.dataset_utils import get_dataset_path, read_dataset
from src.models.model_utils import get_model_path
from src.solver.fischetti import insert_output_constraints_fischetti
from src.solver.tjeng import insert_tjeng_output_constraints
from src.solver.milp import build_network
from src.explication.box import box_relax_input_bounds, box_has_solution


def print_explication(data_index, feature_columns, explanation):
    print(f'Explication for data {data_index}: {feature_columns[explanation].to_list()}')


def minimal_explication(mdl, bounds, method, network_input, network_output, layers, use_box=False):
    mdl = mdl.clone()
    number_classes = len(bounds['output'])
    variables = {
        'output': [mdl.get_var_by_name(f'o_{index}') for index in range(number_classes)],
        'binary': mdl.binary_var_list(number_classes - 1, name='b')
    }
    input_constraints = mdl.add_constraints(
        [mdl.get_var_by_name(f'x_{index}') == feature for index, feature in enumerate(network_input)],
        names='input')
    mdl.add_constraint(mdl.sum(variables['binary']) >= 1)
    if method == 'fischetti':
        insert_output_constraints_fischetti(mdl, network_output, variables)
    else:
        insert_tjeng_output_constraints(mdl, bounds['output'], network_output, variables)
    relax_input_mask = np.zeros_like(network_input, dtype=bool)
    for index, constraint in enumerate(input_constraints):
        mdl.remove_constraint(constraint)
        relax_input_mask[index] = True
        if use_box:
            input_bounds = box_relax_input_bounds(network_input, bounds['input'], relax_input_mask)
            if box_has_solution(input_bounds, layers, network_output):
                print(f'Feature {index} is not relevant (using box)')
                continue
        mdl.solve(log_output=False)
        if mdl.solution is not None:
            mdl.add_constraint(constraint)
            relax_input_mask[index] = False
            print(f'Feature {index} is relevant (using solver)')
            continue
        print(f'Feature {index} is not relevant (using solver)')
    return ~relax_input_mask


def get_minimal_explication(dataset_name, method, use_box=True, number_samples=None):
    train_data = read_dataset(get_dataset_path(dataset_name, 'train.csv'))
    validation_data = read_dataset(get_dataset_path(dataset_name, 'validation.csv'))
    test_data = read_dataset(get_dataset_path(dataset_name, 'test.csv'))
    number_samples = len(test_data) if not number_samples or number_samples > len(test_data) else number_samples
    features = test_data.columns[:-1]
    dataframe = pd.concat([train_data, validation_data, test_data], ignore_index=True)
    model = load_model(get_model_path(dataset_name, f'{dataset_name}.h5'))
    layers = model.layers
    mdl, bounds = build_network(layers, dataframe, method)
    for data_index, data in test_data.iterrows():
        if data_index == number_samples:
            break
        print(f'Getting explication for data {data_index}...')
        network_input = data.iloc[:-1]
        network_output = np.argmax(model.predict(tf.reshape(network_input, (1, -1))))
        explanation = \
            minimal_explication(mdl, bounds, method, network_input, network_output, layers, use_box)
        print_explication(data_index, features, explanation)
