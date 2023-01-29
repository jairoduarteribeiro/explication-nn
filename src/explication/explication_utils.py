from src.solver.fischetti import insert_output_constraints_fischetti
from src.solver.tjeng import insert_tjeng_output_constraints


def print_explication(data_index, feature_columns, explanation):
    columns = [int(constraint.lhs.name[2:]) for constraint in explanation]
    result = list(feature_columns[columns])
    print(f'Explication for data {data_index}: {result}')


def get_minimal_explication(mdl, bounds, method, network_input, network_output):
    mdl = mdl.clone()
    number_classes = len(bounds['output'])
    variables = {
        'output': [mdl.get_var_by_name(f'o_{index}') for index in range(number_classes)],
        'binary': mdl.binary_var_list(number_classes - 1, name='b')
    }
    input_constraints = mdl.add_constraints(
        [mdl.get_var_by_name(f'x_{index}') == feature.numpy() for index, feature in enumerate(network_input[0])],
        names='input')
    mdl.add_constraint(mdl.sum(variables['binary']) >= 1)
    if method == 'fischetti':
        insert_output_constraints_fischetti(mdl, network_output, variables)
    else:
        insert_tjeng_output_constraints(mdl, bounds['output'], network_output, variables)
    for index, constraint in enumerate(input_constraints):
        mdl.remove_constraint(constraint)
        mdl.solve(log_output=False)
        if mdl.solution is not None:
            mdl.add_constraint(constraint)
            print(f'Feature {index} is relevant (using solver)')
            continue
        print(f'Feature {index} is not relevant (using solver)')
    return mdl.find_matching_linear_constraints('input')
