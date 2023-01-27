import numpy as np
from src.solver.solver_utils import maximize, minimize


def build_fischetti_network(mdl, layers, variables):
    output_bounds = []
    number_layers = len(layers)
    for layer_index, layer in enumerate(layers):
        x = variables['input'] if layer_index == 0 else variables['intermediate'][layer_index - 1]
        _A = layer.get_weights()[0].T
        _b = layer.bias.numpy()
        number_neurons = len(_A)
        if layer_index != number_layers - 1:
            _s = variables['auxiliary'][layer_index]
            _a = variables['decision'][layer_index]
            _y = variables['intermediate'][layer_index]
        else:
            _s = np.empty(number_neurons)
            _a = np.empty(number_neurons)
            _y = variables['output']
        for neuron_index, (A, b, y, a, s) in enumerate(zip(_A, _b, _y, _a, _s)):
            result = A @ x + b
            if layer_index != number_layers - 1:
                mdl.add_constraint(result == y - s, ctname=f'c_{layer_index}_{neuron_index}')
                mdl.add_indicator(a, y <= 0, 1)
                mdl.add_indicator(a, s <= 0, 0)
                upper_bound_y = maximize(mdl, y)
                upper_bound_s = maximize(mdl, s)
                y.set_ub(upper_bound_y)
                s.set_ub(upper_bound_s)
            else:
                mdl.add_constraint(result == y, ctname=f'c_{layer_index}_{neuron_index}')
                upper_bound = maximize(mdl, y)
                lower_bound = minimize(mdl, y)
                y.set_ub(upper_bound)
                y.set_lb(lower_bound)
                output_bounds.append([lower_bound, upper_bound])
    return mdl, output_bounds


def insert_output_constraints_fischetti(mdl, network_output, variables):
    output_variable = variables['output'][network_output]
    binary_index = 0
    for output_index, output in enumerate(variables['output']):
        if output_index != network_output:
            indicator = variables['binary'][binary_index]
            mdl.add_indicator(indicator, output_variable <= output, 1)
            binary_index += 1
