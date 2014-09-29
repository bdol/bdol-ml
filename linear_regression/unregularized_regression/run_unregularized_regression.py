"""
Learns unregularized linear regression weights using stochastic gradient
descent.

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
from data_utils import split_train_test, RMSE
from linear_regression import LinearRegression
from sklearn import preprocessing
from sklearn.datasets import fetch_california_housing


print "Loading data..."
housing = fetch_california_housing(data_home='/home/bdol/data')
train_data, test_data, train_target, test_target = split_train_test(
    housing.data, housing.target
)

# Normalize the data
train_data = preprocessing.scale(train_data)
test_data = preprocessing.scale(test_data)

# Append bias feature
train_data = np.hstack((train_data, np.ones((train_data.shape[0], 1),
                                            dtype=train_data.dtype)))
test_data = np.hstack((test_data, np.ones((test_data.shape[0], 1),
                                          dtype=test_data.dtype)))

train_target = train_target[:, None]
test_target = test_target[:, None]

lin_reg = LinearRegression()
print "Training..."
lin_reg.train(train_data, train_target)
print
print "Done!"

# Get training error
train_preds = lin_reg.test(train_data)
test_preds = lin_reg.test(test_data)
print "Train error:", RMSE(train_preds, train_target)
print "Test error:", RMSE(test_preds, test_target)
