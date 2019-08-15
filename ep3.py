import numpy as np

def logistic_fit(X, y, w = None, batch_size = None, learning_rate = 1e-2,
                 num_iterations = 1000, return_history = False):

    sigmoid = lambda z:  1 / (1 + np.exp(-z))

    N = X.shape[0]

    X = np.c_[np.ones(N), X]

    if w == None:
        w = np.random.random(X.shape[1])

    if batch_size == None or batch_size > N:
        batch_size = N

    indices = np.random.permutation(N)

    cost_history = np.zeros(num_iterations)
        
    for it in range(num_iterations):
        X = X[indices]
        y = y[indices]
        for i in range(0, N, batch_size):
            X_i = X[i:i + batch_size]
            y_i = y[i:i + batch_size]
            pred = sigmoid(X_i@w)
            w -= (1/N)*learning_rate*(X_i.T@(pred - y_i))
            if return_history:
                pred = sigmoid(X_i@w)
                cost_history[it] += (1/batch_size)*(-1*y_i.T@np.log(pred) - (1 - y_i).T@np.log(1 - pred))

    if return_history:
        return w, cost_history
    return w

def predict(X, w):
    sigmoid = lambda z:  1 / (1 + np.exp(-z))
    X = np.c_[np.ones(X.shape[0]), X]
    return sigmoid(X@w)