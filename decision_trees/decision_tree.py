"""
A single decision tree. Includes functions to both train and test on given
numerical data. The decision tree evaluates 10 splits for each feature
(spaced linearly between the min/max feature values) and splits based on
entropy. For an example on how to run this tree with MNIST,
see run_decision_tree.py

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

"""d
A single decision tree. Includes functions to both train and test on given
numerical data. This implementation has the following properties:

    Split function:
    Impurity measure:


"""



from interface_utils import prog_bar
from ml_functions import entropy
import numpy as np


class Node():
    def __init__(self):
        self.left = None
        self.right = None
        self.terminal = False
        self.split_feature = []
        self.split_val = []
        self.target_value = []


class DecisionTree():
    def __init__(self, max_depth):
        self.max_depth = max_depth

    def _choose_feature(self, train_data, train_target, x_range,
                        remaining_features):
        py = np.mean(train_target, axis=0)
        H = entropy(py)

        max_ig = -1
        split_feature = -1
        best_split_val = -1

        t = 1
        for x_i in remaining_features:
            prog_bar(t, len(remaining_features))
            t += 1

            # Here, we take 10 splits over the range of features, each of
            # which we will evaluate
            split_vals = np.linspace(np.min(x_range[x_i]),
                                     np.max(x_range[x_i]),
                                     11)

            # If there are no informative splits, then skip this feature
            if split_vals[0] == split_vals[-1]:
                continue

            # In addition, remove the last split because all values will be
            # less than or equal to it
            split_vals = split_vals[0:-1]

            for split_val in split_vals:
                split = (train_data[:, x_i] <= split_val).astype(float)

                sum_x = np.sum(split, axis=0).astype(float)
                sum_notx = train_data.shape[0] - sum_x

                py_given_x = np.zeros((train_target.shape[1], 1))
                py_given_notx = np.zeros((train_target.shape[1], 1))

                for y in range(train_target.shape[1]):
                    y_given_x = ((split==1) & (train_target[:, y]==1))
                    y_given_notx = ((split==0) & (train_target[:, y]==1))
                    py_given_x[y] = np.sum(y_given_x) / sum_x
                    py_given_notx[y] = np.sum(y_given_notx) / sum_notx


                # Compute the conditional entropy and information gain
                px = np.mean(split)
                cond_H = px * entropy(py_given_x) + (1 - px) * entropy(py_given_notx)
                ig = H - cond_H

                if ig > max_ig:
                    max_ig = ig
                    split_feature = x_i
                    best_split_val = split_val

        return split_feature, best_split_val, max_ig

    """
    This is the recursive function we use to iteratively build the tree.
    """

    def _split_node(self, train_data, train_target, x_range,
                    default_value, remaining_features,
                    depth):

        node = Node()

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
                        train_target.shape[0] <= 1:
            node.terminal = True

            if train_target.shape[0] == 0:
                node.target_value = default_value
            else:
                node.target_value = np.mean(train_target, 0)

            print "*** depth: {0} [{1}]: Leaf predictions: {2}".format(
                depth, train_target.shape[0], node.target_value)
            return node

        print "Splitting at depth {0}...".format(depth)
        node.split_feature, node.split_val, max_ig = self._choose_feature(
            train_data, train_target, x_range, remaining_features)

        node.target_value = np.mean(train_target, 0)
        remaining_features = np.delete(remaining_features,
                                       np.where(
                                           remaining_features == node.split_feature))

        leftidx = train_data[:, node.split_feature] <= node.split_val
        rightidx = train_data[:, node.split_feature] > node.split_val

        print "depth: {0} [{1}]: Split on feature {2}. L/R = {3}/{4}".format(
            depth, train_target.shape[0], node.split_feature, np.sum(leftidx),
            np.sum(rightidx))

        node.left = self._split_node(train_data[leftidx, :],
                                     train_target[leftidx],
                                     x_range,
                                     node.target_value,
                                     remaining_features,
                                     depth + 1)

        node.right = self._split_node(train_data[rightidx, :],
                                      train_target[rightidx],
                                      x_range,
                                      node.target_value,
                                      remaining_features,
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
        x_range = []
        for j in range(0, train_data.shape[1]):
            x_range.append(np.unique(train_data[:, j]))

        return self._split_node(train_data,
                                train_target,
                                x_range,
                                np.mean(train_target.astype(float), axis=0),
                                np.arange(0, train_data.shape[1]),
                                0)

    def test(self, root, test_data, test_target):
        errs = 0.0
        for i in range(0, test_data.shape[0]):
            yhat = np.argmax(self._dt_value(root, test_data[i, :]))
            if test_target[i, yhat] != 1:
                errs += 1.0

        return errs/test_data.shape[0]