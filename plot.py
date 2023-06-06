from matplotlib import pyplot as plt
import pandas as pd
import os
import numpy as np
from my_linear_regression import MyLinearRegression as MyLR


def plot(model):
    # Plotting the data
    data = pd.read_csv('data.csv')
    # print(data.head)
    X = np.array(data['km']).reshape(-1, 1)
    y = np.array(data['price']).reshape(-1, 1)
    plt.scatter(X, y, color='blue', label='Data')

    # Predicting the output using the trained model
    y_pred = model.predict_(X)

    # Plotting the linear regression line
    plt.plot(X, y_pred, color='red', label='Linear Regression')

    # Calculating the precision (R-squared value)
    SSR = np.sum((y_pred - np.mean(y))**2)
    SST = np.sum((y - np.mean(y))**2)
    R_squared = SSR / SST

    # Displaying the precision (R-squared value)
    plt.title('Linear Regression')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.legend()
    plt.show()

    print('Precision (R-squared value):', R_squared)


model = MyLR(thetas=np.array([[0], [0]]).astype('float64'), alpha=1e-3,
             max_iter=50000, normalize='y')


if model.check_model() is False:
    print('the model does not exist, so cannot plot')

else:
    model.load_model()
    # print('hello')
    plot(model)
