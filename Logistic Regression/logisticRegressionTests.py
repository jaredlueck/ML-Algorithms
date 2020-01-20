from unittest import TestCase
from logisticRegression import LogisticRegression
import numpy as np
from numpy.testing import assert_array_equal


class LogisticRegressionTests(TestCase):

    def setUp(self):
        self.x = np.array([
            [1, 1.1],
            [1, 1.3],
            [1, 1.7],
            [2, 1.2]
        ])

        self.y = np.array([
            0, 0, 1, 1
        ])

        self.theta = np.array([
            1.1, 2.2
        ])

        self.model = LogisticRegression(0.1)

    def test_compute_cost(self):
        cost, grad = self.model.compute_cost(self.theta, self.x, self.y)
        self.assertEqual(round(cost, 4), 1.886)
        assert_array_equal(np.around(grad, 4), [0.0482, 0.058])


