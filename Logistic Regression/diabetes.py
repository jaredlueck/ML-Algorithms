import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from logisticRegression import LogisticRegression
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


if __name__ == "__main__":
    data = pd.read_csv('diabetes2.csv')
    y = data['Outcome']

    x = data.drop(columns=['Outcome'])
    columns = x.columns
    min_max_scaler = preprocessing.MinMaxScaler()
    x = min_max_scaler.fit_transform(x)
    x = pd.DataFrame(x, columns=columns)

    x.insert(0, 'X_0', np.ones(x.shape[0]))

    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1)

    model = LogisticRegression(0.001)

    iterations, cost = model.fit(X_train.values, y_train.values.reshape(-1, 1), 2, 10000)

    plot_cost_vs_iterations(iterations, cost[0])

    predictions = model.predict(X_test.values)

    predictions = np.rint(predictions)

    test_accuracy = accuracy_score(y_test, predictions)

    print(test_accuracy)







