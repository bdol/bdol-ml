"""
This is the same as the original decision tree class, except the
choose_feature function has been optimized. The original function, although
easy to read, was too slow to use in random forests. I have memoized some
parts (namely the split values) and vectorized the feature computation.

In addition, I added an additional parameter for use in random forests,
a feature subsampling percentage to consider at each split.

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
from ml_functions import safe_entropy, safe_plogp
from py_utils import *
import numpy as np
import operator
from copy import deepcopy

class Node():
    def __init__(self):
        self.left = None
        self.right = None
        self.terminal = False
        self.split_feature = []
        self.split_val = []
        self.target_value = []


class FastDecisionTree():
    def __init__(self, max_depth, num_splits, feat_subset=1.0, debug=False):
        self.max_depth = max_depth
        self.num_splits = num_splits
        self.feat_subset = feat_subset
        self.debug = debug

    # TODO: this should probably be refactored
    def _choose_feature(self, train_data_original, train_target,
                        x_range_original):

        # Subsample the features, if applicable
        train_data = None
        x_range = dict()
        if self.feat_subset != 1.0:
            r = np.random.rand(train_data_original.shape[1])
            keep_idx = r < self.feat_subset

            train_data = train_data_original[:, keep_idx]
            sorted_keys = sorted(x_range_original.keys())
            for i in range(len(keep_idx)):
                if keep_idx[i]:
                    x_range[sorted_keys[i]] = x_range_original[sorted_keys[i]]

        else:
            train_data = train_data_original
            x_range = x_range_original

        sorted_x_range = sorted(x_range.items(), key=operator.itemgetter(0))

        py = np.mean(train_target, axis=0)
        H = safe_entropy(py[:, None])

        max_ig = -1
        split_feature = -1
        best_split_val = -1

        t = 1
        for s in range(self.num_splits):

            splits = [l[1][s] for l in sorted_x_range]
            split = (train_data <= splits)

            sum_x = (np.sum(split, axis=0)).astype(float)
            sum_notx = train_data.shape[0] - sum_x

            py_given_x = np.zeros((train_target.shape[1], len(x_range)))
            py_given_notx = np.zeros((train_target.shape[1], len(x_range)))

            for y in range(train_target.shape[1]):
                if self.debug:
                    prog_bar(t, self.num_splits*train_target.shape[1])
                    t += 1

                y_given_x = (split & (train_target[:, y]==1)[:, None])
                y_given_notx = ((split == False) & (train_target[:, y]==1)[:,
                                                   None])

                y_given_x_sum = (np.sum(y_given_x, axis=0)).astype(float)
                y_given_notx_sum = (np.sum(y_given_notx, axis=0)).astype(float)

                py_given_x[y, :] = y_given_x_sum / sum_x
                py_given_notx[y, :] = y_given_notx_sum / sum_notx

            px = np.mean(split, axis=0)
            cond_H = px * safe_entropy(py_given_x) + (1 - px) * safe_entropy(py_given_notx)
            ig = H - cond_H
            ig_max = np.max(ig)
            ig_argmax = np.argmax(ig)

            if ig_max > max_ig:
                max_ig = ig_max
                split_feature = sorted_x_range[ig_argmax][0]
                best_split_val = splits[ig_argmax]

        return split_feature, best_split_val, max_ig

    """
    This is the recursive function we use to iteratively build the tree.
    """

    def _split_node(self, train_data, train_target, remaining_features,
                    default_value, depth):

        node = Node()

        # Figure out the ranges of the remaining features

        # We need to check various conditions here to see if we should stop
        # training this branch. Specifically if one of the conditions is met,
        # we stop:
        # 1) We are at the maximum depth
        # 2) There are no more features to split on
        # 3) All of the examples have the same label
        # 4) There are no examples at this node (return the default value)
        py = np.mean(train_target, axis=0)
        if depth == self.max_depth or \
                        len(remaining_features) == 0 or \
                        np.max(py) == 1 or \
                        train_data.shape[0] <= 1:
            node.terminal = True

            if train_target.shape[0] == 0:
                node.target_value = default_value
            else:
                node.target_value = np.mean(train_target, 0)

            if self.debug:
                print "*** depth: {0} [{1}]: Leaf predictions: {2}".format(
                    depth, train_target.shape[0], node.target_value)
            return node

        x_range = dict()
        for j in range(len(remaining_features)):
            min_val = np.min(train_data[:, j], axis=0)
            max_val = np.max(train_data[:, j], axis=0)
            x_range[remaining_features[j]] = np.linspace(min_val,
                                                         max_val,
                                                         self.num_splits,
                                                         endpoint=False)
        if self.debug:
            print "Splitting at depth {0}...".format(depth)

        node.split_feature, node.split_val, max_ig = self._choose_feature(
            train_data, train_target, x_range)

        node.target_value = np.mean(train_target, 0)

        # Remove the feature from consideration and from the training data
        data_split_feature = remaining_features.index(node.split_feature)
        remaining_features_copy = deepcopy(remaining_features)
        remaining_features_copy.remove(node.split_feature)
        mask = np.ones(train_data.shape[1], dtype=bool)
        mask[data_split_feature] = False
        train_data = train_data[:, mask]

        leftidx = train_data[:, data_split_feature] <= node.split_val
        rightidx = train_data[:, data_split_feature] > node.split_val

        if self.debug:
            print "depth: {0} [{1}]: Split on feature {2}. L/R = {3}/{4}".format(
                depth, train_target.shape[0], node.split_feature, np.sum(leftidx),
                np.sum(rightidx))

        node.left = self._split_node(train_data[leftidx, :],
                                     train_target[leftidx],
                                     remaining_features_copy,
                                     node.target_value,
                                     depth + 1)

        node.right = self._split_node(train_data[rightidx, :],
                                      train_target[rightidx],
                                      remaining_features_copy,
                                      node.target_value,
                                      depth + 1)

        return node

    def _dt_value(self, root, x):
        node = root
        while not node.terminal:
            if x[node.split_feature] <= node.split_val:
                node = node.left
            else:
                node = node.right

        return node.target_value

    def train(self, train_data, train_target):
        # Compute the range of values for each feature
        M = train_data.shape[1]

        # Here, we only consider informative features (not a single value
        # over all examples). We store these informative features as a tuple
        # containing the feature index and the feature splits. Therefore
        # we don't need the remaining_features list.
        mask = np.ones(M, dtype=bool)
        remaining_features = []
        for j in range(0, M):
            vals = np.unique(train_data[:, j])
            if vals[0] != vals[-1]:
                remaining_features.append(j)
            else:
                mask[j] = False
        train_data_mod = np.copy(train_data[:, mask])

        if self.debug:
            print "Removed {0} uninformative features (out of {1}).".format(
                M-len(remaining_features), M
            )

        return self._split_node(train_data_mod,
                                train_target,
                                remaining_features,
                                np.mean(train_target, axis=0),
                                0)

    def test(self, root, test_data, test_target):
        errs = 0.0
        for i in range(0, test_data.shape[0]):
            yhat = np.argmax(self._dt_value(root, test_data[i, :]))
            if test_target[i, yhat] != 1:
                errs += 1.0

        return errs/test_data.shape[0]

    def test_preds(self, root, test_data, test_target):
        yhat = np.zeros((test_data.shape[0], 1))
        for i in range(0, test_data.shape[0]):
            yhat[i] = np.argmax(self._dt_value(root, test_data[i, :]))

        return yhat
