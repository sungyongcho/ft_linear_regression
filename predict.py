from my_linear_regression import MyLinearRegression as MyLR
import os

directory = './'  # Replace with the actual directory path

# Get all files in the directory
files = os.listdir(directory)

# Print the list of files


def load_model():
    file_exists = False
    for file in files:
        if (file == 'model'):
            print('A Model file exists.')
            print(file)
            file_exists = True
    if file_exists is False:
        print('A Model file DOES NOT exist.')


def prompt():
    mileage = input("Enter a mileage:")


load_model()
