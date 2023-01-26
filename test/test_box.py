import unittest
from numpy.testing import assert_array_equal
from src.explication.box import box_fix_input_bounds, box_forward, box_check_solution


class TestBox(unittest.TestCase):
    def test_box_fix_input_bounds(self):
        network_inputs = [1.0, 2.0, 3.0, 4.0]
        input_bounds = [2.5, 3.5]
        input_index = 2
        fixed_bounds = box_fix_input_bounds(network_inputs, input_bounds, input_index)
        assert_array_equal(fixed_bounds, [[1.0, 1.0], [2.0, 2.0], [2.5, 3.5], [4.0, 4.0]])

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


if __name__ == '__main__':
    unittest.main()
