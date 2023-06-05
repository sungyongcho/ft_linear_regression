import pandas as pd
import os
import numpy as np
from my_linear_regression import MyLinearRegression as MyLR

directory = './'  # Replace with the actual directory path

# Get all files in the directory
files = os.listdir(directory)

# Print the list of files


def train_again_or_not():
    print("A Model already exists, do you want to train them again?")
    while True:
        answer = input("Enter 'Y' for Yes or 'N' for No: ").upper()
        if answer == "Y":
            print("You chose Yes.")
            return True
        elif answer == "N":
            print("You chose No.")
            return False
        else:
            print("Invalid input. Please enter 'Y' or 'N'.")


def check_existing_model():
    for file in files:
        if (file == 'model'):
            return True
    return False


def create_model():
    data = pd.read_csv('data.csv')
    # print(data.head)
    X = np.array(data['km']).reshape(-1, 1)
    y = np.array(data['price']).reshape(-1, 1)
    model = MyLR(thetas=np.array([[0], [0]]).astype(
        'float64'), alpha=1e-3, max_iter=50000, normalize='y')

    model.fit_(X, y)
    print(model.predict_((np.array(float(240000)).reshape(-1, 1))))


if __name__ == "__main__":
    create_model()
    # if (check_existing_model() is True):
    #     if (train_again_or_not()):
    #         print('train')
    #     else:
    #         print('not train')
