from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from my_linear_regression import MyLinearRegression as MyLR

model = MyLR(thetas=np.array([[0], [0]]).astype(
    'float64'), alpha=1e-3, max_iter=1000, normalize='y')

# if model.check_model() is False:
#     print('The model does not exist, so cannot plot')
# else:
#     # model.load_model()

data = pd.read_csv('data.csv')
X = np.array(data['km']).reshape(-1, 1)
y = np.array(data['price']).reshape(-1, 1)

# Create a figure and axis
fig, ax = plt.subplots()

# Set up the scatter plot
scatter = ax.scatter(X, y, color='blue', label='Data')

# Set up the initial line
x_line = np.array([np.min(X), np.max(X)])
y_line = model.thetas[1] * x_line + model.thetas[0]  # Use zero theta here
line, = ax.plot([], [], color='red', label='Linear Regression')

# Initialize text annotation
annotation = ax.annotate('', (0.5, -0.1), xycoords='axes fraction',
                         ha='center', va='center')

ax.set_ylim(np.min(y)-1000, np.max(y)+1000)


# Initialization function for the animation
def init():
    line.set_data([], [])
    annotation.set_text('')
    # r2_text.set_text('R2 Score: ')
    return line, annotation


def update(frame):
    normalized, not_normalized = model.fit_(X, y)
    model.thetas = not_normalized
    # theta_history.append(model.copy())
    y_line = normalized[1] * x_line + normalized[0]
    # print(x_line.flatten(), y_line.flatten())
    # valid_points = np.where(y_line != 0)
    # x_line_valid = x_line[valid_points]
    # y_line_valid = y_line[valid_points]

    line.set_data(x_line, y_line)

    # Calculate MSE and R2 score
    y_pred = model.predict_(X)
    mse = model.mse_(y, y_pred)
    # r2 = model.r2_score_(y, y_pred)

    y_pred = model.predict_(X)
    mse = np.mean((y - y_pred) ** 2)
    annotation.set_text(f'MSE: {mse:.2f}, R2 Score:')

    return line, annotation


# Create the animation
animation = FuncAnimation(fig, update, frames=50,
                          init_func=init, blit=True)

# Display the animation
plt.title('Linear Regression Animation')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.show()
