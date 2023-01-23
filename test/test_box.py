import unittest
from numpy.testing import assert_array_equal
from src.box import box_forward


class TestBox(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
