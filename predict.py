import pickle

import numpy as np
from my_linear_regression import MyLinearRegression as MyLR
import os

directory = './'  # Replace with the actual directory path

# Get all files in the directory
files = os.listdir(directory)


def check_model():
    file_exists = False
    for file in files:
        if (file == 'model'):
            print('A Model file exists.')
            file_exists = True
    if file_exists is False:
        print('A Model file DOES NOT exist.')
    return file_exists


def load_model(model_file='model'):
    try:
        with open(model_file, 'rb') as f:
            theta = pickle.load(f)
        print("The model has been loaded from:", model_file)
        print("Thetas", theta.flatten())
        return theta
    except FileNotFoundError:
        print("Model file not found. Please provide a valid model file.")
    except pickle.UnpicklingError:
        print("Error loading the model. The model file may be corrupted.")


model = MyLR(thetas=np.array([[0], [0]]).astype('float64'), alpha=1e-3,
             max_iter=50000, normalize='y')

if model.check_model() is True:
    model.load_model()

mileage = input("Enter a mileage: ")

pred_val = model.predict_((np.array(float(mileage)).reshape(-1, 1))).item()
print("The predicted value is:", pred_val)
