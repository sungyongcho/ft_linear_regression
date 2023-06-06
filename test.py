from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from my_linear_regression import MyLinearRegression as MyLR

model = MyLR(thetas=np.array([[0], [0]]).astype(
    'float64'), alpha=1e-3, max_iter=10000, normalize='y')

# if model.check_model() is False:
#     print('The model does not exist, so cannot plot')
# else:
#     # model.load_model()

data = pd.read_csv('data.csv')
X = np.array(data['km']).reshape(-1, 1)
y = np.array(data['price']).reshape(-1, 1)
x_target, y_target = model.normalize_(X, y)


normalized, not_normalized = model.fit_(X, y)
model.thetas = not_normalized

normalized, not_normalized = model.fit_(X, y)
model.thetas = not_normalized

normalized, not_normalized = model.fit_(X, y)
model.thetas = not_normalized

normalized, not_normalized = model.fit_(X, y)
model.thetas = not_normalized

normalized, not_normalized = model.fit_(X, y)
