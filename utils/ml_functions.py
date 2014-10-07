"""
A set of ML-related functions that are used in a variety of models.

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

"""
A common function used to compute the entropy. This is a "safe" entropy
function in that invalid values (NaN or inf) are clamped to 0. This is used
in the decision tree module where dividing by 0 could happen.
"""
def safe_plogp(x):
    e = x * np.log2(x)
    if hasattr(e, "__len__"):
        e[np.isinf(e)] = 0
        e[np.isnan(e)] = 0
    else:
        if np.isnan(e) or np.isinf(e):
            e = 0.0

    return e

"""
The standard entropy function, using the "safe" plogp function which clamps
invalid inputs to 0.
"""
def safe_entropy(x):
    return -np.sum(safe_plogp(x))

def marginal_entropy(x):
    return -np.sum(x * np.log2(x))

def safe_binary_entropy(x):
    l_px = np.log2(x)
    l_pnotx = np.log2(1-x)
    if hasattr(x, "__len__"):
        l_px[np.isinf(l_px)] = 0
        l_pnotx[np.isinf(l_pnotx)] = 0
    else:
        if np.isnan(l_px) or np.isinf(l_px):
            l_px = 0
        if np.isnan(l_pnotx) or np.isinf(l_pnotx):
            l_pnotx = 0

    return -(np.multiply(x, l_px)
              + np.multiply((1-x), l_pnotx))
