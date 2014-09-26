"""
An implementation of a random forest. Uses the provided DecisionTree class.
The RF is parameterized by the following values:

    1. n_trees: The number of trees in the forest
    2. boot_percent: The bootstrap percent. For each tree, we select a
    bootstrap sample of the dataset (with replacement). This is the
    percentage of the dataset to use for each tree.
    3. feat_percent: The percentage of features to consider at each split.
    Instead of examining all features at each split, we select feat_percent
    of them to consider.

In addition, you need to specify the typical decision tree parameters,
namely max depth and the number of linear split points.

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
from fast_decision_tree import FastDecisionTree
from scipy import stats
import numpy as np

class RandomForest:
    def __init__(self, n_trees, max_depth, num_splits,
        boot_percent=0.3, feat_percent=0.3, threaded=False, debug=False):

        self.n_trees = n_trees
        self.max_depth = max_depth
        self.num_splits = num_splits
        self.boot_percent = boot_percent
        self.feat_percent = feat_percent
        self.threaded = threaded
        self.debug = debug

        self.roots = []

    def train(self, train_data, train_target):
        t = 0
        for i in range(self.n_trees):
            if self.debug:
                prog_bar(t, self.n_trees)
                t += 1


            keep_idx = np.random.rand(train_data.shape[0]) <= \
                       self.boot_percent

            boot_train_data = train_data[keep_idx, :]
            boot_train_target = train_target[keep_idx]

            dt = FastDecisionTree(self.max_depth, self.num_splits,
                                  feat_subset=self.feat_percent)

            r = dt.train(boot_train_data, boot_train_target)
            self.roots.append(r)

        if self.debug:
            prog_bar(self.n_trees, self.n_trees)

    def test(self, test_data, test_target):
        t = 0
        # TODO: refactor the RF test function to depend not on an external
        # root but on itself
        dt = FastDecisionTree(1, 1)
        yhat_forest = np.zeros((test_data.shape[0], self.n_trees))
        for i in range(len(self.roots)):
            r = self.roots[i]
            if self.debug:
                prog_bar(t, self.n_trees)
                t += 1

            yhat_forest[:, 0:] = dt.test_preds(r, test_data, test_target)

        if self.debug:
            prog_bar(self.n_trees, self.n_trees)
            print

        yhat = stats.mode(yhat_forest, axis=1)[0]
        return yhat
