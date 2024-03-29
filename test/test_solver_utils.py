import unittest

import pandas as pd
from docplex.mp.model import Model

from src.solver.solver_utils import get_input_domain_and_bounds, get_input_variables, \
    get_intermediate_variables, get_decision_variables, get_output_variables


class TestSolverUtils(unittest.TestCase):
    def test_get_input_domain_and_bounds(self):
        dataframe = pd.DataFrame({
            'age': [18, 19, 20],
            'weight': [65.9, 58.0, 89.5],
            'gender': [0, 1, 0]
        })
        domain, bounds = get_input_domain_and_bounds(dataframe)
        self.assertEqual(domain, ['I', 'C', 'B'])
        self.assertEqual(bounds, [(18, 20), (58.0, 89.5), (0, 1)])

    def test_get_input_variables(self):
        domain = ['I', 'C', 'B']
        bounds = [(18, 20), (58.0, 89.5), (0, 1)]
        mdl = Model()
        input_variables = get_input_variables(mdl, domain, bounds)
        self.assertTrue(input_variables[0].is_integer())
        self.assertEqual(input_variables[0].name, 'x_0')
        self.assertEqual(input_variables[0].lb, bounds[0][0])
        self.assertEqual(input_variables[0].ub, bounds[0][1])
        self.assertTrue(input_variables[1].is_continuous())
        self.assertEqual(input_variables[1].name, 'x_1')
        self.assertEqual(input_variables[1].lb, bounds[1][0])
        self.assertEqual(input_variables[1].ub, bounds[1][1])
        self.assertTrue(input_variables[2].is_binary())
        self.assertEqual(input_variables[2].name, 'x_2')
        self.assertEqual(input_variables[2].lb, bounds[2][0])
        self.assertEqual(input_variables[2].ub, bounds[2][1])

    def test_get_intermediate_variables(self):
        number_variables = 3
        mdl = Model()
        intermediate_variables = get_intermediate_variables(mdl, 1, number_variables)
        self.assertEqual(len(intermediate_variables), number_variables)
        self.assertTrue(intermediate_variables[0].is_continuous())
        self.assertEqual(intermediate_variables[0].name, 'y_1_0')
        self.assertEqual(intermediate_variables[0].lb, 0)
        self.assertEqual(intermediate_variables[0].ub, mdl.infinity)
        self.assertTrue(intermediate_variables[1].is_continuous())
        self.assertEqual(intermediate_variables[1].name, 'y_1_1')
        self.assertEqual(intermediate_variables[1].lb, 0)
        self.assertEqual(intermediate_variables[1].ub, mdl.infinity)
        self.assertTrue(intermediate_variables[2].is_continuous())
        self.assertEqual(intermediate_variables[2].name, 'y_1_2')
        self.assertEqual(intermediate_variables[2].lb, 0)
        self.assertEqual(intermediate_variables[2].ub, mdl.infinity)

    def test_get_decision_variables(self):
        number_variables = 3
        mdl = Model()
        decision_variables = get_decision_variables(mdl, 1, number_variables)
        self.assertEqual(len(decision_variables), number_variables)
        self.assertTrue(decision_variables[0].is_binary())
        self.assertEqual(decision_variables[0].name, 'a_1_0')
        self.assertEqual(decision_variables[0].lb, 0)
        self.assertEqual(decision_variables[0].ub, 1)
        self.assertTrue(decision_variables[1].is_binary())
        self.assertEqual(decision_variables[1].name, 'a_1_1')
        self.assertEqual(decision_variables[1].lb, 0)
        self.assertEqual(decision_variables[1].ub, 1)
        self.assertTrue(decision_variables[2].is_binary())
        self.assertEqual(decision_variables[2].name, 'a_1_2')
        self.assertEqual(decision_variables[2].lb, 0)
        self.assertEqual(decision_variables[2].ub, 1)

    def test_get_output_variables(self):
        number_variables = 3
        mdl = Model()
        output_variables = get_output_variables(mdl, number_variables)
        self.assertEqual(len(output_variables), number_variables)
        self.assertTrue(output_variables[0].is_continuous())
        self.assertEqual(output_variables[0].name, 'o_0')
        self.assertEqual(output_variables[0].lb, -mdl.infinity)
        self.assertEqual(output_variables[0].ub, mdl.infinity)
        self.assertTrue(output_variables[1].is_continuous())
        self.assertEqual(output_variables[1].name, 'o_1')
        self.assertEqual(output_variables[1].lb, -mdl.infinity)
        self.assertEqual(output_variables[1].ub, mdl.infinity)
        self.assertTrue(output_variables[2].is_continuous())
        self.assertEqual(output_variables[2].name, 'o_2')
        self.assertEqual(output_variables[2].lb, -mdl.infinity)
        self.assertEqual(output_variables[2].ub, mdl.infinity)


if __name__ == '__main__':
    unittest.main()
