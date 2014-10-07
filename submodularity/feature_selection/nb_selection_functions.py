"""
A set of a hand-tuned functions to do feature selection for fully-factorized
(Naive Bayes) and unfactorized distributions.

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
from ml_functions import safe_entropy, safe_binary_entropy
import itertools
import numpy as np

"""
Submodular feature selection for a Naive Bayes model. NB treats each feature
independently, so the notion of "context," a benefit of submodular
optimization, is ignored here. Instead, the information gain of each feature
is calculated independently, and the top-k features are selected.
"""
def nb_feature_selection(X, Y, k):
    H = safe_binary_entropy(np.mean(Y))

    px = X.mean(axis=0)
    sum_x = X.sum(axis=0).astype(float)
    sum_notx = X.shape[0] - sum_x

    y_given_x = ((X == 1).todense() & (Y == 1)[:, None])
    y_given_notx = ((X != 1).todense() & (Y == 1)[:, None])

    py_given_x = y_given_x.sum(axis=0) / sum_x
    py_given_notx = y_given_notx.sum(axis=0) / sum_notx

    # Correct for 0 counts
    py_given_x[np.isnan(py_given_x)] = 0.0
    py_given_notx[np.isnan(py_given_notx)] = 0.0

    cond_H = np.multiply(px, safe_binary_entropy(py_given_x)) \
             + np.multiply(1 - px, safe_binary_entropy(py_given_notx))
    IG = (H - cond_H)
    best_features = IG.argsort().tolist()[0]
    best_features.reverse()
    # A = set(best_features[0:k])
    A = best_features[0:k]

    return A

"""
This function computes the marginal IG for the element e in the context of
X_A. It does not do any sorting
"""
def _initialize_lazy_bounds(X, Y):
    H = safe_binary_entropy(np.mean(Y))

    px = X.mean(axis=0)
    sum_x = X.sum(axis=0).astype(float)
    sum_notx = X.shape[0] - sum_x

    y_given_x = ((X == 1).todense() & (Y == 1)[:, None])
    y_given_notx = ((X != 1).todense() & (Y == 1)[:, None])

    py_given_x = y_given_x.sum(axis=0) / sum_x
    py_given_notx = y_given_notx.sum(axis=0) / sum_notx

    # Correct for 0 counts
    py_given_x[np.isnan(py_given_x)] = 0.0
    py_given_notx[np.isnan(py_given_notx)] = 0.0

    cond_H = np.multiply(px, safe_binary_entropy(py_given_x)) \
             + np.multiply(1 - px, safe_binary_entropy(py_given_notx))
    return (H - cond_H)

def _get_lazy_bound(X, Y, A, e):
    IG = safe_binary_entropy(np.mean(Y))

    # The new set with e added to the context
    S = set(A)
    S.add(e)
    S_ind = list(S)
    X_A = X[:, S_ind]

    x_settings = ["".join(x) for x in itertools.product("01", repeat =
                                                            (len(S_ind)))]
    for x in x_settings:
        x_vals = [int(v) for v in x]
        mask = np.ones((X.shape[0], 1), dtype=bool)
        for i in range(len(x_vals)):
            mask = mask & (X_A[:, i] == x_vals[i]).todense()

        px = np.mean(mask)
        if px > 0:
            sum_x = mask.sum().astype(float)
            y_given_x = mask & (Y == 1)[:, None]
            py_given_x = y_given_x.sum(axis=0) / sum_x

            cond_H = np.multiply(px, safe_binary_entropy(py_given_x))

            IG -= cond_H

    return IG[0, 0]

def _resort_rho(rho):
    ep = rho.pop(0)
    for i in range(0, len(rho)):
        if ep[1] > rho[i][1]:
            rho.insert(i, ep)
            return


"""
"Unfactorized" feature selection, a true submodular (rather than
just modular) optimization problem. In this case there are interactions between
each of the individual features, so that adding a new feature takes into
account this context. In this case, we use the greedy algorithm to
approximate the true top-k feature set.

The main problem with IG with unfactorized features is that evaluating F(A)
is itself exponential (we must sum over all x in X_A), and is intractable to do
naively. However, we can use the lazy greedy algorithm (Minoux 1978) to speed up
the computation. In addition, many values of x in X_A do not appear, so we can
also safely ignore those.
"""
def unfactorized_feature_selection(X, Y, k):
    A = set()

    # Set up the marginal gains queue
    # Of the form [index, marginal gain, |S_{i-1}|
    bounds = _initialize_lazy_bounds(X, Y)
    idx = bounds.argsort().tolist()[0]
    idx.reverse()
    vals = bounds.tolist()[0]
    vals.sort(reverse=True)
    rho = zip(idx, vals, [0]*len(idx))

    ep = rho.pop(0)
    A.add(ep[0])
    last_F = ep[1]
    print "***** Added", ep[0], " with marginal gain", last_F

    # Begin the feature selection routine
    for i in range(1, k):
        # Extract the first element from the queue and recompute the bound
        new_max = -1
        next_max = rho[1][1]
        while new_max < next_max:
            if rho[0][2] == len(A):
                ep = rho.pop(0)
                A.add(ep[0])
                last_F += ep[1]
                print "***** Added", ep[0], " with marginal gain", ep[1]
                continue

            next_max = rho[1][1]
            old_max = rho[0][1]
            jawn = _get_lazy_bound(X, Y, A, rho[0][0])
            new_max = jawn-last_F
            rho[0] = (rho[0][0], new_max, len(A))

            # We need to update the queue in this case
            if new_max < next_max:
                _resort_rho(rho)
                continue

            ep = rho.pop(0)
            A.add(ep[0])
            last_F += ep[1]
            print "***** Added", ep[0], " with marginal gain", ep[1]

    return A
