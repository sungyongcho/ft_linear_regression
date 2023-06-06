import pickle
import numpy as np
import matplotlib.pyplot as plt


class MyLinearRegression():
    """
    Description:
    My personnal linear regression class to fit like a boss.
    """

    def __init__(self, thetas, alpha=1e-3, max_iter=1000, normalize='f'):
        if isinstance(thetas, tuple):
            thetas = np.array(thetas, dtype=np.float64)
        elif not isinstance(thetas, np.ndarray):
            thetas = np.array(thetas, dtype=np.float64)
        elif not thetas.dtype is np.float64:
            thetas = np.array(thetas, dtype=np.float64)
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas
        self.normalize = normalize

    def simple_gradient(self, x, y, theta):
        """Computes a gradient vector from three non-empty numpy.array, without any for loop.
        The three arrays must have compatible shapes.
        Args:
        x: has to be a numpy.array, a matrix of shape m * 1.
        y: has to be a numpy.array, a vector of shape m * 1.
        theta: has to be a numpy.array, a 2 * 1 vector.
        Return:
        The gradient as a numpy.ndarray, a vector of dimension 2 * 1.
        None if x, y, or theta is an empty numpy.ndarray.
        None if x, y and theta do not have compatible dimensions.
        Raises:
        This function should not raise any Exception.
        """
        if x.size == 0 or y.size == 0 or theta.size == 0:
            return None
        if x.shape[0] != y.shape[0] or theta.shape != (2, 1):
            return None

        m = x.shape[0]
        xi = np.hstack((np.ones((m, 1)), x))
        hypothesis = np.dot(xi, theta)
        gradient = np.dot(xi.T, hypothesis - y) / m

        return gradient

    def gradient(self, x, y):
        """
        Computes a gradient vector from three non-empty numpy.array, without any for-loop.
        The three arrays must have the compatible dimensions.
        (Update: thetas are being taken from the class member var.)

        Args:
        x: has to be an numpy.array, a matrix of dimension m * n.
        y: has to be an numpy.array, a vector of dimension m * 1.

        Return:
        The gradient as a numpy.array, a vector of dimensions n * 1,
        containg the result of the formula for all j.
        None if x, y, or theta are empty numpy.array.
        None if x, y and theta do not have compatible dimensions.
        None if x, y or theta is not of expected type.

        Raises:
        This function should not raise any Exception.
        """
        m = len(y)  # Number of training examples

        # Add a column of ones to X as the first column
        x_prime = np.concatenate((np.ones((m, 1)), x), axis=1)

        # Compute the difference between predicted and actual values
        diff = np.dot(x_prime, self.thetas) - y

        # Compute the gradient
        gradient = (1/m) * np.dot(x_prime.T, diff)

        return gradient

    def fit_(self, x, y):
        """
        Description:
        Fits the model to the training dataset contained in x and y.
        Args:
        x: has to be a numpy.array, a matrix of dimension m * n:
        (number of training examples, number of features).
        y: has to be a numpy.array, a vector of dimension m * 1:
        (number of training examples, 1).
        theta: has to be a numpy.array, a vector of dimension (n + 1) * 1:
        (number of features + 1, 1).
        alpha: has to be a float, the learning rate
        max_iter: has to be an int, the number of iterations done during the gradient descent
        Return:
        new_theta: numpy.array, a vector of dimension (number of features + 1, 1).
        None if there is a matching dimension problem.
        None if x, y, theta, alpha or max_iter is not of expected type.
        Raises:
        This function should not raise any Exception.
        """
        self.thetas = np.array([[0], [0]]).astype('float64')
        if self.normalize == 'y':
            x_target, y_target = self.normalize_(x, y)
        else:
            x_target = x
            y_target = y

        m = len(y)  # Number of training examples
        n = x.shape[1]  # Number of features

        if (x.shape[0] != m) or (self.thetas.shape[0] != (n + 1)):
            return None

        for i in range(self.max_iter):
            gradient_update = self.gradient(x_target, y_target)
            if gradient_update is None:
                return None
            # Convert gradient_update to float64
            theta_before = self.thetas.copy()
            self.thetas -= self.alpha * gradient_update.astype(np.float64)
            if (i % 10000 == 0):
                print(i, "th:", np.hstack(self.thetas))
                print("Gain:", self.gain_(theta_before, self.thetas))

        if self.normalize == 'y':
            print('before normalization', self.thetas.flatten())
            self.thetas = self.denormalize_theta(self.thetas, x, y)
            print('after normalization', self.thetas.flatten())
        return self.thetas

    def gain_(self, theta_before, theta_after):
        """
        Calculates the gain or improvement achieved by the model after an update in the theta values.
        Args:
        theta_before: numpy.array, vector of shape (n + 1, 1) representing the theta values before the update.
        theta_after: numpy.array, vector of shape (n + 1, 1) representing the theta values after the update.
        Returns:
        gain: numpy.array, vector of shape (n + 1, 1) representing the gain in theta values.
        None if theta_before or theta_after is an empty numpy array.
        None if theta_before and theta_after do not have compatible dimensions.
        """
        if theta_before.size == 0 or theta_after.size == 0:
            return None
        if theta_before.shape != theta_after.shape:
            return None

        return np.sum(np.abs(theta_after - theta_before))

    def predict_(self, x):
        """Computes the prediction vector y_hat from two non-empty numpy.array.
        Args:
        x: has to be an numpy.array, a vector of dimensions m * n.
        theta: has to be an numpy.array, a vector of dimensions (n + 1) * 1.
        Return:
        y_hat as a numpy.array, a vector of dimensions m * 1.
        None if x or theta are empty numpy.array.
        None if x or theta dimensions are not appropriate.
        None if x or theta is not of expected type.
        Raises:
        This function should not raise any Exception.
        """
        xp = np.hstack((np.ones((x.shape[0], 1)), x))
        y_hat = np.dot(xp, self.thetas)
        return y_hat

    # def simple_predict(self, x):
    #     new = np.zeros(shape=x.shape)
    #     for i, item in enumerate(x):
    #         new[i] = self.thetas[0] + self.thetas[1] * item
    #     return new

    def loss_elem_(self, y, y_hat):
        """
        Description:
        Calculates all the elements (y_pred - y)^2 of the loss function.
        Args:
        y: has to be an numpy.array, a vector.
        y_hat: has to be an numpy.array, a vector.
        Returns:
        J_elem: numpy.array, a vector of dimension (number of the training examples,1).
        None if there is a dimension matching problem between X, Y or theta.
        None if any argument is not of the expected type.
        Raises:
        This function should not raise any Exception.
        """
        a = y_hat - y
        return (a ** 2)
        # print(np.sqrt(a))

    def loss_(self, y, y_hat):
        """
        Description:
        Calculates the value of loss function.
        Args:
        y: has to be an numpy.array, a vector.
        y_hat: has to be an numpy.array, a vector.
        Returns:
        J_value : has to be a float.
        None if there is a dimension matching problem between X, Y or theta.
        None if any argument is not of the expected type.
        Raises:
        This function should not raise any Exception.
        """

        a = self.loss_elem_(y, y_hat)

        return np.sum(a)/len(a) / 2

    def mse_elem(self, y, y_hat):
        a = y_hat - y
        return (a ** 2)

    def mse_(self, y, y_hat):
        """
        Description:
        Calculate the MSE between the predicted output and the real output.
        Args:
        y: has to be a numpy.array, a vector of dimension m * 1.
        y_hat: has to be a numpy.array, a vector of dimension m * 1.
        Returns:
        mse: has to be a float.
        None if there is a matching dimension problem.
        Raises:
        This function should not raise any Exceptions.
        """
        if y.shape != y_hat.shape:
            return None
        return np.mean(np.square(y_hat - y))

    def minmax_(self, x):
        """Computes the normalized version of a non-empty numpy.ndarray using the min-max standardization.
        Args:
        x: has to be an numpy.ndarray, a vector.
        Returns:
        x’ as a numpy.ndarray.
        None if x is a non-empty numpy.ndarray or not a numpy.ndarray.
        Raises:
        This function shouldn’t raise any Exception.
        """

        min_x = np.min(x)
        max_x = np.max(x)

        if min_x == max_x:
            return x

        return (x - min_x) / (max_x - min_x)

    def normalize_(self, x, y):
        """
        Normalize the feature matrix x by subtracting the mean and dividing by the standard deviation.
        Args:
        x: numpy.array, a matrix of shape m * n (number of training examples, number of features).
        Returns:
        x_normalized: numpy.array, a matrix of shape m * n with normalized feature values.
        None if x is an empty numpy.array.
        """
        if x.size == 0:
            return None

        if x.size == 0 or y.size == 0:
            return None

        X_normalized = self.minmax_(x)
        y_normalized = self.minmax_(y)

        return X_normalized, y_normalized

    def denormalize_theta(self, theta, x, y):
        x_min = np.min(x)
        x_max = np.max(x)
        y_min = np.min(y)
        y_max = np.max(y)

        theta_denorm = np.zeros_like(theta)
        theta_denorm[0] = theta[0] * (y_max - y_min) + y_min
        theta_denorm[1:] = theta[1:] * (y_max - y_min) / (x_max - x_min)

        return theta_denorm

    def save_model(self, output_file='model'):
        with open(output_file, 'wb') as f:
            pickle.dump(self.thetas, f)
        print("The model has been saved in: ",
              output_file, "in same directory")
