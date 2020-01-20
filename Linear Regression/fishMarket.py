import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
from linearRegression import LinearRegression
import matplotlib.pyplot as plt


def plot_regression(x, y, y_pred):
    """ Method to plot the model in 2 dimensions
    Parameters:
    x: values for x-axis
    y: values for y-axis """
    plt.scatter(x, y)
    plt.plot(x, y_pred)
    plt.xlabel('Reduced Features')
    plt.ylabel('Weights(g)')
    plt.title('Fish Market Species Weights')
    plt.show()


def plot_cost_vs_iterations(iterations, cost):
    """ Plot cost curve.
    Parameters:
    iterations: array containing x-values
    cost: array containing y-values """
    plt.plot(iterations, cost)
    plt.xlabel('iterations')
    plt.ylabel('cost')
    plt.title('Cost vs Iterations')
    plt.show()


if __name__ == '__main__':
    # Read data from CSV file
    data = pd.read_csv('Fish.csv')

    # Replace categorical Species column with dummy encoding for 7 species e.g. [0, 0, 0, 1, 0, 0, 0]
    dummy_species = pd.get_dummies(data['Species'])
    x = data.drop(columns=['Weight', 'Species'])
    x = pd.concat([dummy_species, x], axis=1)
    columns = x.columns
    y = data['Weight']

    # Principle component analysis to transform data to 2D for visualization
    pca = PCA(n_components=1)
    x_1d = pca.fit_transform(x)

    # Feature scaling for 1D and multivariate data sets
    min_max_scaler = preprocessing.MinMaxScaler()
    x = min_max_scaler.fit_transform(x)
    x = pd.DataFrame(x, columns=columns)

    x_1d = min_max_scaler.fit_transform(x_1d)

    # Create column of ones to act as x0 (y_int) to simplify operations
    x_0 = np.ones(shape=data.shape[0])
    x.insert(0, 'X_0', x_0)
    x_1d = pd.DataFrame({'X_1': x_1d.reshape(159)})
    x_1d.insert(0, 'X_0', x_0)

    # Splitting multivariate data
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1)

    model = LinearRegression(0.1)

    iterations, cost = model.fit(X_train.values, y_train.values, 1000)

    # Plot the cost vs iterations curve to verify gradient descent is working and
    # to help chose a number of iterations
    plot_cost_vs_iterations(iterations, cost)

    predictions = model.predict(X_test.values)

    train_predictions = model.predict(X_train.values)

    results = pd.DataFrame({'predictions': predictions, 'actual': y_test})

    results.to_csv('predictions.csv')

    # Compute 2 different errors; mean squared error and mean absolute error
    test_mean_squared_error = mean_squared_error(y_test, predictions)

    test_mean_abs_error = mean_absolute_error(y_test, predictions)

    train_mean_squared_error = mean_squared_error(y_train, train_predictions)

    train_mean_abs_error = mean_absolute_error(y_train, train_predictions)

    print("Mean squared error test set: %f" % test_mean_squared_error)

    print("Mean absolute error test set: %f" % test_mean_abs_error)

    print("Mean squared error training set: %f" % train_mean_squared_error)

    print("Mean absolute error training set: %f" % train_mean_abs_error)

    # This is part is for reduced dimension version for visualization
    X_train, X_test, y_train, y_test = train_test_split(x_1d, y, random_state=1)

    model = LinearRegression(0.1)

    iterations, cost = model.fit(X_train.values, y_train.values, 1000)

    plot_cost_vs_iterations(iterations, cost)

    predictions = model.predict(X_test.values)

    # drop x_0 since we need only one feature to have the form y=mx+b
    x_plot = X_test.drop(columns=['X_0'])

    plot_regression(x_plot.values, y_test.values, predictions)




