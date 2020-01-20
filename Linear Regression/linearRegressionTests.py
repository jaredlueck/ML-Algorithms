from unittest import TestCase
from linearRegression import LinearRegression
import numpy as np
from numpy.testing import assert_array_equal


class LinearRegressionTests(TestCase):

    def setUp(self):
        self.x = np.array([
            [1, 1.1, 1.3, 1.4],
            [1, 1.1, 1.5, 2.4],
            [1, 1.7, 1.5, 2.2],
            [2, 1.2, 1.6, 2.1]
        ])

        self.y = np.array([
            3.1, 3.3, 3.9, 4.1
        ])

        self.theta = np.array([
            1.1, 1.2, 2.0, 2.1
        ])

        self.model = LinearRegression(0.1)

    def test_compute_cost(self):
        cost, grad = self.model.compute_cost(self.theta, self.x, self.y)
        self.assertEqual(round(cost, 4), 21.6334)
        assert_array_equal(np.around(grad, 4), [0.8295,  0.8366,   0.9697,  1.3524])
