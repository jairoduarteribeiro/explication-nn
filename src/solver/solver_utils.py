import numpy as np
from docplex.mp.model import Model


def get_input_domain_and_bounds(dataframe):
    domain = []
    bounds = []
    for column in dataframe.columns[:-1]:
        unique_values = dataframe[column].unique()
        if len(unique_values) == 2:
            domain.append('B')
        elif np.any(unique_values.astype(np.int64) != unique_values.astype(np.float64)):
            domain.append('C')
        else:
            domain.append('I')
        lower_bound = dataframe[column].min()
        upper_bound = dataframe[column].max()
        bounds.append([lower_bound, upper_bound])
    return domain, bounds


def get_input_variables(input_domain, input_bounds):
    with Model() as model:
        input_variables = []
        for idx, (domain, bounds) in enumerate(zip(input_domain, input_bounds)):
            lower_bound, upper_bound = bounds
            name = f'x_{idx}'
            if domain == 'C':
                input_variables.append(model.continuous_var(lb=lower_bound, ub=upper_bound, name=name))
            elif domain == 'I':
                input_variables.append(model.integer_var(lb=lower_bound, ub=upper_bound, name=name))
            else:
                input_variables.append(model.binary_var(name=name))
        return input_variables


def get_intermediate_variables(layer_index, number_variables):
    with Model() as model:
        return model.continuous_var_list(number_variables, lb=0, name='y', key_format=f'_{layer_index}_%s')


def get_auxiliary_variables(layer_index, number_variables):
    with Model() as model:
        return model.continuous_var_list(number_variables, lb=0, name='s', key_format=f'_{layer_index}_%s')


def get_decision_variables(layer_index, number_variables):
    with Model() as model:
        return model.binary_var_list(number_variables, name='a', key_format=f'_{layer_index}_%s')


def get_output_variables(number_variables):
    with Model() as model:
        return model.continuous_var_list(number_variables, lb=-model.infinity, name='o')
