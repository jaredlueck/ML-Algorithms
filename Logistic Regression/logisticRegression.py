import numpy as np
from numpy import transpose


class LogisticRegression:

    def __init__(self, learning_rate):
        self.theta = None
        self.learning_rate = learning_rate
        self.reg_param = 0.8

    def compute_cost(self, theta, x, y):
        """Computes the cost and gradient of the current model.

        Parameters:
        X: mxn matrix containing training examples
        y: mx1 vector containing training example y values
        theta: nx1 vector containing model parameters

        Returns:
        cost[]: (n_labels)x1 vector containing computed costs for each label for this iteration
        grad: nx(n_labels) matrix where each column (i) corresponds to the gradients for the ith label"""

        m = len(y)
        h = self.sigmoid(np.matmul(x, theta))

        grad = np.zeros(theta.shape)

        costs = []
        for i in range(0, theta.shape[1]):
            y_tmp = np.array(y == i).astype(int).reshape(-1, 1)
            cost = (1 / m) * (np.matmul(transpose(-y_tmp), np.log(h[:, i])) + np.matmul(transpose(-(1 - y_tmp)), np.log(1 - h[:, i])))
            costs.append(cost)
            grad_i = (self.learning_rate/m) * \
            np.matmul(transpose(x), (h[:, i].reshape(-1, 1) - y_tmp)) + \
            ((self.reg_param/m)*self.theta[:, i].reshape(-1, 1))
            grad[:, i] = grad_i[:, 0]
        return costs, grad

    def fit(self, x, y, n_labels, n_iterations):
        """Fits the model to the provided dataset.
        Parameters:
        x: mxn matrix containing training examples.
        y: mx1 matrix containing correct output for training examples.
        n_labels: number of labels in this dataset
        n_iterations: number of iterations of gradient descent to perform.

        Returns:
        iterations_plot: array containing iterations array for plotting.
        cost_plot: (n_labels)x(n_iterations) matrix where cost_plot[i][j]
        is the computed cost for the ith label on the jth iteration"""

        self.theta = np.zeros(shape=(x.shape[1], n_labels))
        iterations_plot = []
        cost_plot = [[] for i in range(0, n_labels)]
        for i in range(1, n_iterations+1):
            costs, grad = self.compute_cost(self.theta, x, y)
            self.theta = self.theta - grad

            iterations_plot.append(i)

            for j in range(0, n_labels):
                cost_plot[j].append(costs[j])

        return iterations_plot, cost_plot

    def predict(self, x):
        """Generates predictions using fit parameters theta.

        Parameters:
        x: mxn features matrix for predictions

        Returns:
        h: mx1 matrix containing hypothesis"""

        # Theta is a nx(n_labels) matrix
        h = np.argmax(self.sigmoid(np.matmul(x, self.theta)), axis=1)

        return h

    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))

















