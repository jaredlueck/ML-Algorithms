import numpy as np


class SVM:

    def fit(self, x, y, n_iterations):
        m = len(x)
        self.theta = np.zeros(x.shape[1])

