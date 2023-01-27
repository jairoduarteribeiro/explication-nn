import numpy as np
from docplex.mp.model import Model
from src.solver.solver_utils import maximize, minimize


def build_tjeng_network(layers, variables):
    with Model() as model:
        output_bounds = []
        number_layers = len(layers)
        for layer_index, layer in enumerate(layers):
            x = variables['input'] if layer_index == 0 else variables['intermediate'][layer_index - 1]
            _A = layer.get_weights()[0].T
            _b = layer.bias.numpy()
            number_neurons = len(_A)
            if layer_index != number_layers - 1:
                _a = variables['decision'][layer_index]
                _y = variables['intermediate'][layer_index]
            else:
                _a = np.empty(number_neurons)
                _y = variables['output']
            for neuron_index, (A, b, y, a) in enumerate(zip(_A, _b, _y, _a)):
                result = A @ x + b
                upper_bound = maximize(model, result)
                if upper_bound <= 0 and layer_index != number_layers - 1:
                    model.add_constraint(y == 0, ctname=f'c_{layer_index}_{neuron_index}')
                    continue
                lower_bound = minimize(model, result)
                if lower_bound >= 0 and layer_index != number_layers - 1:
                    model.add_constraint(result == y, ctname=f'c_{layer_index}_{neuron_index}')
                    continue
                if layer_index != number_layers - 1:
                    model.add_constraint(y <= result - lower_bound * (1 - a))
                    model.add_constraint(y >= result)
                    model.add_constraint(y <= upper_bound * a)
                else:
                    model.add_constraint(result == y)
                    output_bounds.append([lower_bound, upper_bound])
        return model, output_bounds


def insert_tjeng_output_constraints(model, output_bounds, network_output, variables):
    output_variable = variables['output'][network_output]
    diffs = output_bounds[network_output][1] - np.array(output_bounds)[:, 0]
    binary_index = 0
    for output_index, output in enumerate(variables['output']):
        if output_index != network_output:
            diff = diffs[output_index]
            binary_variable = variables['binary'][binary_index]
            model.add_constraint(output_variable - output - diff * (1 - binary_variable) <= 0)
            binary_index += 1
