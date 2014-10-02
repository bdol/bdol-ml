"""
Example that uses ridge regression on

==============
Copyright Info
==============
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Copyright Brian Dolhansky 2014
bdolmail@gmail.com
"""

import numpy as np
from data_utils import cross_validation_folds, split_train_test, RMSE
from linear_regression import LinearRegression
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.datasets import fetch_california_housing
import sys

# Here we create simulated data to show the benefit of ridge regression
# From Tibshirani 2013, "Modern regression 1: Ridge Regression"
# http://www.stat.cmu.edu/~ryantibs/datamining/lectures/16-modr1.pdf
#
# Our model is a linear function of the features, with 10 large features (
# between 0.5 and 1.0) and 20 smaller ones (between 0.0 and 0.1)
true_w = 0.3*np.random.rand(20, 1)
true_w = np.append(true_w, 0.5*np.random.rand(10, 1) + 0.5)

# Now generate the dataset using the true weights
N = 50
train_data = np.random.rand(N, 30)
train_target = train_data.dot(true_w)[:, None]+np.random.randn(N, 1)
test_data = np.random.rand(N, 30)
test_target = test_data.dot(true_w)[:, None]+np.random.randn(N, 1)

lam_range = np.logspace(-1, 1, 100)
unreg_results = np.zeros((len(lam_range), 1))
reg_results = np.zeros((len(lam_range), 1))

lin_reg = LinearRegression()
i = 0
for l in lam_range:
    lin_reg.train_closed_form_unregularized(train_data, train_target)
    yhat = lin_reg.test(test_data)
    unreg_results[i] = RMSE(yhat, test_target)

    lin_reg.train_closed_form_ridge(train_data, train_target, l)
    yhat = lin_reg.test(test_data)
    reg_results[i] = RMSE(yhat, test_target)

    i += 1

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xscale("log")
ax.plot(lam_range, unreg_results, label="Linear Regression")
ax.plot(lam_range, reg_results, 'r', label="Ridge Regression")
ax.set_title("Unregularized vs. Ridge Regression, RMSE")
ax.set_xlabel("Lambda")
ax.legend()

plt.show()
