import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model

from src.datasets.dataset_utils import get_dataset_path, read_dataset
from src.explication.box import box_relax_input_bounds, box_has_solution
from src.models.model_utils import get_model_path
from src.solver.milp import build_network
from src.solver.tjeng import insert_tjeng_output_constraints


def print_explication(data_index, feature_columns, explanation):
    print(f'Explication for data {data_index}: {np.array(feature_columns)[explanation]}')


def minimal_explication(mdl, bounds, network_input, network_output, layers, features, use_box=False):
    mdl = mdl.clone()
    number_classes = len(bounds['output'])
    variables = {
        'output': [mdl.get_var_by_name(f'o_{index}') for index in range(number_classes)],
        'binary': mdl.binary_var_list(number_classes - 1, name='b')
    }
    input_constraints = mdl.add_constraints(
        [mdl.get_var_by_name(f'x_{index}') == feature for index, feature in enumerate(network_input)], names='input')
    mdl.add_constraint(mdl.sum(variables['binary']) >= 1)
    insert_tjeng_output_constraints(mdl, bounds['output'], network_output, variables)
    relax_input_mask = np.zeros_like(network_input, dtype=bool)
    for index, (constraint, feature) in enumerate(zip(input_constraints, features)):
        mdl.remove_constraint(constraint)
        relax_input_mask[index] = True
        if use_box:
            input_bounds = box_relax_input_bounds(network_input, bounds['input'], relax_input_mask)
            if box_has_solution(input_bounds, layers, network_output):
                print(f'Feature {feature} is not relevant (using box)')
                continue
        mdl.solve(log_output=False)
        if mdl.solution is not None:
            mdl.add_constraint(constraint)
            relax_input_mask[index] = False
            print(f'Feature {feature} is relevant (using solver)')
            continue
        print(f'Feature {feature} is not relevant (using solver)')
    return ~relax_input_mask


def get_minimal_explication(dataset_name, use_box=True, number_samples=None):
    train_data = read_dataset(get_dataset_path(dataset_name, 'train.csv'))
    validation_data = read_dataset(get_dataset_path(dataset_name, 'validation.csv'))
    test_data = read_dataset(get_dataset_path(dataset_name, 'test.csv'))
    dataframe = pd.concat([train_data, validation_data, test_data], ignore_index=True)
    features = list(dataframe.columns)[:-1]
    model = load_model(get_model_path(dataset_name, f'{dataset_name}.h5'))
    layers = model.layers
    mdl, bounds = build_network(layers, dataframe)
    if number_samples:
        test_data = test_data.head(number_samples)
    for data_index, data in test_data.iterrows():
        print(f'Getting explication for data {data_index}...')
        network_input = data.iloc[:-1]
        network_output = np.argmax(model.predict(tf.reshape(network_input, (1, -1))))
        explanation = minimal_explication(mdl, bounds, network_input, network_output, layers, features, use_box)
        print_explication(data_index, features, explanation)
