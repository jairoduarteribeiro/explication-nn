import numpy as np


def box_fix_input_bounds(input_bounds, network_inputs, input_mask):
    network_inputs = np.reshape(network_inputs, (-1, 1))
    network_inputs = np.repeat(network_inputs, 2, axis=1)
    fixed_bounds = np.array(input_bounds)
    fixed_bounds[input_mask] = network_inputs[input_mask]
    return fixed_bounds


def box_forward(input_bounds, input_weights, input_biases, apply_relu=True):
    flipped_bounds = np.flip(input_bounds, axis=1)
    input_bounds = np.concatenate((input_bounds, flipped_bounds), axis=0)
    weights_left = np.array(input_weights)
    weights_left[weights_left < 0] = 0
    weights_right = np.array(input_weights)
    weights_right[weights_right >= 0] = 0
    input_weights = np.concatenate((weights_left, weights_right), axis=1)
    input_biases = np.reshape(input_biases, (-1, 1))
    output_bounds = np.dot(input_weights, input_bounds) + input_biases
    return np.maximum(output_bounds, 0) if apply_relu else output_bounds


def box_check_solution(output_bounds, network_output):
    lower_bound = output_bounds[network_output][0]
    output_bounds = np.delete(output_bounds, network_output, axis=0)
    max_upper_bound = np.max(output_bounds, axis=0)[1]
    return lower_bound > max_upper_bound
