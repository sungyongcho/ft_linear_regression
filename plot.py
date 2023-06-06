from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
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
    # plot(model)

data = pd.read_csv('data.csv')
# print(data.head)
X = np.array(data['km']).reshape(-1, 1)
y = np.array(data['price']).reshape(-1, 1)
# Assuming you have the following variables: X (input features), y (target variable), and model (trained linear regression model)

# Create a figure and axis
fig, ax = plt.subplots()

# Set up the scatter plot
scatter = ax.scatter(X, y, color='blue', label='Data')

# Set up the initial line
x_line = np.array([np.min(X), np.max(X)])
y_line = model.predict_(np.zeros((2, 1)))  # Use zero theta here
line, = ax.plot(x_line, y_line, color='red', label='Linear Regression')
# Initialization function for the animation


def init():
    line.set_data(x_line, y_line)
    return line,

# Update function for the animation


def update(frame):
    # Calculate the predicted values for the current frame
    model.max_iter = 1
    model.thetas = model.fit_(X, y)
    print("theta:", model.thetas)
    y_pred = model.predict_(X[frame].reshape(-1, 1))

    # Update the line data
    line.set_data(x_line, y_pred)

    return line,


# Create the animation
animation = FuncAnimation(fig, update, frames=500,
                          init_func=init, blit=True)

# Display the animation
plt.title('Linear Regression Animation')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.show()
