import numpy as np


def get_input_domain_and_bounds(dataframe):
    domain = []
    bounds = []
    columns = list(dataframe.columns)
    for column in columns:
        unique_values = dataframe[column].unique()
        if len(unique_values) == 2:
            domain.append('B')
        elif np.any(unique_values.astype(np.int64) != unique_values.astype(np.float64)):
            domain.append('C')
        else:
            domain.append('I')
        lower_bound = dataframe[column].min()
        upper_bound = dataframe[column].max()
        bounds.append((lower_bound, upper_bound))
    return domain, bounds


def get_input_variables(mdl, input_domain, input_bounds):
    input_variables = []
    for index, (domain, bounds) in enumerate(zip(input_domain, input_bounds)):
        lower_bound, upper_bound = bounds
        name = f'x_{index}'
        if domain == 'C':
            input_variables.append(mdl.continuous_var(lb=lower_bound, ub=upper_bound, name=name))
        elif domain == 'I':
            input_variables.append(mdl.integer_var(lb=lower_bound, ub=upper_bound, name=name))
        else:
            input_variables.append(mdl.binary_var(name=name))
    return input_variables


def get_intermediate_variables(mdl, layer_index, number_variables):
    return mdl.continuous_var_list(number_variables, lb=0, name='y', key_format=f'_{layer_index}_%s')


def get_decision_variables(mdl, layer_index, number_variables):
    return mdl.binary_var_list(number_variables, name='a', key_format=f'_{layer_index}_%s')


def get_output_variables(mdl, number_variables):
    return mdl.continuous_var_list(number_variables, lb=-mdl.infinity, name='o')


def maximize(mdl, variable):
    mdl.maximize(variable)
    mdl.solve()
    objective = mdl.solution.get_objective_value()
    mdl.remove_objective()
    return objective


def minimize(mdl, variable):
    mdl.minimize(variable)
    mdl.solve()
    objective = mdl.solution.get_objective_value()
    mdl.remove_objective()
    return objective
