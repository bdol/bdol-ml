"""
A set of ML-related functions that are used in a variety of models.

Brian Dolhansky, 24-09-2014
bdolmail@gmail.com
"""

import numpy as np

"""
A common function used to compute the entropy.
"""
def plogp(x):
    e = x * np.log2(x)
    # Set values outside the range of log to 0
    e[np.isinf(e)] = 0
    e[np.isnan(e)] = 0
    return e

"""
The standard entropy function.
"""
def entropy(x):
    return -np.sum(plogp(x), axis=0)