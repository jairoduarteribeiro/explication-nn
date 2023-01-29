import numpy as np


def box_relax_input_bounds(network_inputs, input_bounds, input_mask):
    input_bounds = np.array(input_bounds)
    network_inputs = np.array(network_inputs).reshape((-1, 1))
    relaxed_bounds = np.repeat(network_inputs, 2, axis=1)
    relaxed_bounds[input_mask] = input_bounds[input_mask]
    return relaxed_bounds


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


def box_has_solution(bounds, layers, network_output):
    for layer_index, layer in enumerate(layers):
        weights = layer.get_weights()[0].T
        biases = layer.get_weights()[1]
        bounds = box_forward(bounds, weights, biases) if layer_index != len(layers) - 1 \
            else box_forward(bounds, weights, biases, apply_relu=False)
    return box_check_solution(bounds, network_output)
