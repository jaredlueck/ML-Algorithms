import numpy as np


class LinearRegression:

    def __init__(self, learning_rate):
        self.theta = None
        self.learning_rate = learning_rate

    def compute_cost(self, theta, x, y):
        """Computes the cost and gradient of the current model.

        Parameters:
        X: mxn matrix containing training examples
        y: mx1 vector containing training example y values
        theta: nx1 vector containing model parameters

        Returns:
        cost: cost computed with current parameters
        grad: computed gradient"""

        m = len(y)
        h = np.matmul(x, theta)

        cost = (1/(2*m)) * sum((h-y)**2)

        grad = (self.learning_rate/m) * np.matmul(np.matrix.transpose(x), (h-y))

        return cost, grad

    def fit(self, x, y, n_iterations):
        """Fits the model to the provided dataset.
        Parameters:
        x: mxn matrix containing training examples.
        y: mx1 matrix containing correct output for training examples.
        n_iterations: number of iterations of gradient descent to perform.

        Returns:
        iterations_plot: array containing iterations array for plotting.
        cost_plot: array containing cost array for plotting."""

        self.theta = np.zeros(shape=x.shape[1])
        iterations_plot = []
        cost_plot = []
        for i in range(1, n_iterations+1):
            cost, grad = self.compute_cost(self.theta, x, y)
            self.theta = self.theta - grad
            cost_plot.append(cost)
            iterations_plot.append(i)

        return iterations_plot, cost_plot

    def predict(self, x):
        """Generates predictions using fit parameters theta.

        Parameters:
        x: mxn features matrix for predictions

        Returns:
        h: mx1 matrix containing hypothesis"""

        h = np.matmul(x, self.theta)

        return h

















