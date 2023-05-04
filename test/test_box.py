import unittest

import numpy as np
from numpy.testing import assert_array_equal

from src.solver.box import box_relax_input_bounds, box_forward, box_check_solution, \
    box_has_solution


class Layer:
    def __init__(self, weights, biases):
        self.weights = [np.array(weights), np.array(biases)]

    def get_weights(self):
        return self.weights


class TestBox(unittest.TestCase):
    def test_box_relax_input_bounds(self):
        network_inputs = [1.0, 2.0, 3.0, 4.0]
        input_bounds = [[0.5, 1.5], [1.5, 2.5], [2.5, 3.5], [3.5, 4.5]]
        input_mask = [True, False, True, False]
        relaxed_bounds = box_relax_input_bounds(network_inputs, input_bounds, input_mask)
        assert_array_equal(relaxed_bounds, [[0.5, 1.5], [2.0, 2.0], [2.5, 3.5], [4.0, 4.0]])

    def test_box_forward_with_relu(self):
        input_bounds = [[1, 2], [3, 4]]
        input_weights = [[1, 1], [1, -1], [-1, 1]]
        input_biases = [0, 2, -1]
        output_bounds = box_forward(input_bounds, input_weights, input_biases)
        assert_array_equal(output_bounds, [[4, 6], [0, 1], [0, 2]])

    def test_box_forward_without_relu(self):
        input_bounds = [[1, 2], [3, 4]]
        input_weights = [[1, 1], [1, -1], [-1, 1]]
        input_biases = [0, 2, -1]
        output_bounds = box_forward(input_bounds, input_weights, input_biases, False)
        assert_array_equal(output_bounds, [[4, 6], [-1, 1], [0, 2]])

    def test_box_check_solution(self):
        output_bounds = [[0.1, 0.4], [0.6, 1.4], [-0.6, 0.2]]
        network_output = 1
        has_solution = box_check_solution(output_bounds, network_output)
        self.assertTrue(has_solution)
        output_bounds = [[0.1, 0.7], [0.6, 1.4], [-0.6, 0.2]]
        network_output = 1
        has_solution = box_check_solution(output_bounds, network_output)
        self.assertFalse(has_solution)

    def test_box_has_solution(self):
        bounds = [[0.0, 0.3], [0.1, 0.4]]
        layers = [
            Layer(weights=[[1, 1], [1, -1]], biases=[0.0, 0.0]),
            Layer(weights=[[1, 1], [1, -1]], biases=[0.5, -0.5])
        ]
        network_output = 0
        has_solution = box_has_solution(bounds, layers, network_output)
        self.assertTrue(has_solution)
        bounds = [[0.0, 0.6], [0.1, 0.7]]
        has_solution = box_has_solution(bounds, layers, network_output)
        self.assertFalse(has_solution)


if __name__ == '__main__':
    unittest.main()
