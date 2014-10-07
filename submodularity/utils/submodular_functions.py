"""
Includes various submodular functions used throughout the submodular examples.

Note that all functions are of the form F(X, A, args), where args is a
dictionary containing additional arguments needed by a particular function.

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
from ml_functions import safe_binary_entropy

"""
The information gain of a subset of variables X_A \subset X, defined as:
    F(A) = IG(X_A, Y) = H(Y) - H(Y | X)
Equivalent to:
    \sum_{x_A} p(x_A) H( p(Y | X_A) )

This function assumes X and Y are 0/1 valued, such as for document
classification where we have a dictionary of words as features, and a 0/1
label of whether or not a given document belongs to the target class.

Also note that this function is normalized, i.e. F(null set) = 0.

Ref. Krause, Guestrin (http://submodularity.org/submodularity-slides.pdf)
"""
def information_gain_nb(X, A, args):
    if len(A) == 0:
        return 0.0

    Y = args['Y']
    A_ind = list(A)

    H = safe_binary_entropy(np.mean(Y))

    px = X[:, A_ind].mean(axis=0)
    sum_x = X[:, A_ind].sum(axis=0).astype(float)
    sum_notx = X.shape[0] - sum_x

    y_given_x = ((X[:, A_ind] == 1).todense() & (Y == 1)[:, None])
    y_given_notx = ((X[:, A_ind] != 1).todense() & (Y == 1)[:, None])

    py_given_x = y_given_x.sum(axis=0) / sum_x
    py_given_notx = y_given_notx.sum(axis=0) / sum_notx

    cond_H = np.multiply(px, safe_binary_entropy(py_given_x)) \
             + np.multiply((1-px), safe_binary_entropy(py_given_notx))

    IG = H-cond_H
    return IG.sum()

"""
A simple submodular function which is defined as:
    f(A) = |X_A|
"""
def cardinality(X, A, args):
    return len(A)

