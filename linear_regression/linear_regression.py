"""
A set of linear regression learning functions. This file includes:

    1) Unregularized linear regression
    2) L2-regularized (Tikhonov) regression (Ridge regression)
    3) L1-regularized regression (The Lasso)

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

from interface_utils import prog_bar
from py_utils import exit_with_err
import numpy as np
import sys

class LinearRegression:
    def __init__(self):
        self.w = None

    def _calc_reg(self, w, regularization, lam):
        if regularization == 'None':
            return 0.0
        elif regularization == 'L1':
            return 0.0
        elif regularization == 'L2':
            return lam*w
        else:
            exit_with_err("Error in LinearRegression: Unsupported "
                          "regularization {0}".format(regularization))

    """
    Batch gradient descent with a least-squares loss.
    """
    def _gd(self, train_data, train_target, regularization, lam,
            step_size=1E-5, num_epochs=1000):

        self.w = np.zeros((train_data.shape[1], 1))

        t = 0
        for i in range(0, num_epochs):
            t += 1
            if t % 100 == 0:
                prog_bar(t, num_epochs)

            yhat = np.dot(train_data, self.w)
            grad = np.sum((yhat - train_target)*train_data, axis=0) - \
                   self._calc_reg(self.w, regularization, lam)
            self.w -= step_size*grad[:, None]

        prog_bar(num_epochs, num_epochs)

    def _closed_form(self, train_data, train_target):
        x = train_data
        self.w = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(train_target)

    def train(self, train_data, train_target, regularization='None', lam=0.1):
        self._gd(train_data, train_target, regularization, lam)

    def test(self, test_data):
        if self.w == None:
            exit_with_err("Error in LinearRegression: Weight vector is None, "
                          "did you run train before testing?")

        return np.dot(test_data, self.w)
