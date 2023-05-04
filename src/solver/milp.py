from docplex.mp.model import Model

from src.solver.solver_utils import get_input_domain_and_bounds, get_input_variables, get_output_variables, \
    get_intermediate_variables, get_decision_variables
from src.solver.tjeng import build_tjeng_network


def build_network(layers, dataframe):
    mdl = Model(name='original')
    x = dataframe.iloc[:, :-1]
    input_domain, input_bounds = get_input_domain_and_bounds(x)
    variables = {
        'input': get_input_variables(mdl, input_domain, input_bounds),
        'intermediate': [],
        'decision': []
    }
    for layer_index, layer in enumerate(layers):
        number_variables = layer.get_weights()[0].shape[1]
        if layer_index == len(layers) - 1:
            variables['output'] = get_output_variables(mdl, number_variables)
            break
        variables['intermediate'].append(get_intermediate_variables(mdl, layer_index, number_variables))
        variables['decision'].append(get_decision_variables(mdl, layer_index, number_variables))
    mdl, output_bounds = build_tjeng_network(mdl, layers, variables)
    bounds = {'input': input_bounds, 'output': output_bounds}
    return mdl, bounds
