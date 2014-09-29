import numpy as np
import matplotlib.pyplot as plt

"""
Randomly samples p percent of the given data for train and uses the other 1-p
percent for test. Assumes that the data is NxM, where N is the number of
examples and M is the number of features.
"""
def split_train_test(data, target, p=0.7):
    n = data.shape[0]
    num_train = int(np.floor(p*n))
    train_idx = np.random.choice(n, num_train, replace=False)
    test_idx = np.setdiff1d(np.arange(n), train_idx)

    train_data = data[train_idx, :]
    test_data = data[test_idx, :]
    train_target = target[train_idx]
    test_target = target[test_idx]

    return train_data, test_data, train_target, test_target

"""
Converts integral target values into a NxV indicator matrix, where each row
is an indicator vector of dimension V (if V is the max label). Assumes that
the value "0" is included in the labels.
"""
def integral_to_indicator(integral_target):
    v = int(np.max(integral_target)+1)
    n = integral_target.shape[0]
    y = np.zeros((n, v))
    for i in range(n):
        y[i, int(integral_target[i])] = 1.0

    return y

def RMSE(yhat, y):
    n = yhat.shape[0]
    return np.sqrt(1.0/n*np.sum(np.square(yhat-y)))

"""
For each of the variables specified in vars, plot a 2-D plot of the target
vs. the feature.
"""
def plot_regressors(data, target, vars=None, descr=None):
    if vars == None:
        vars = range(0, data.shape[1])

    for i in vars:
        fig = plt.figure()
        plt.scatter(data[:, i], target)
        if descr == None:
            plt.xlabel("Variable {0}".format(i))
        else:
            plt.xlabel("{0}".format(descr[i]))
        plt.ylabel("Target")
        plt.show()
        plt.close(fig)

