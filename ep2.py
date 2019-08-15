import numpy as np
import time
from util import get_housing_prices_data, r_squared
from plots import plot_points_regression

def normal_equation_weights(X, y):
    """
    Calculates the weights of a linear function using the normal equation method.
    You should add into X a new column with 1s.

    :param X: design matrix
    :type X: np.ndarray(shape=(N, d))
    :param y: regression targets
    :type y: np.ndarray(shape=(N, 1))
    :return: weight vector
    :rtype: np.ndarray(shape=(d+1, 1))
    """
    
    # START OF YOUR CODE:
    X = np.c_[np.ones(X.shape[0]), X]
    w = np.linalg.inv(X.T@X)@X.T@y
    # END YOUR CODE

    return w

def normal_equation_prediction(X, w):
    """
    Calculates the prediction over a set of observations X using the linear function
    characterized by the weight vector w.
    You should add into X a new column with 1s.

    :param X: design matrix
    :type X: np.ndarray(shape=(N, d))
    :param w: weight vector
    :type w: np.ndarray(shape=(d+1, 1))
    :param y: regression prediction
    :type y: np.ndarray(shape=(N, 1))
    """
    
    # START OF YOUR CODE:
    X = np.c_[np.ones(X.shape[0]), X]
    prediction = X@w
    # END YOUR CODE
    
    return prediction