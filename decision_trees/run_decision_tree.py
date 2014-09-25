"""
An example that runs a single decision tree using MNIST. This single tree can
achieve ~20% error rate on a random 70/30 train/test split on the original MNIST
data (with a depth limit of 10).

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

from decision_tree import DecisionTree
from fast_decision_tree import FastDecisionTree
from sklearn.datasets import fetch_mldata
from data_utils import integral_to_indicator, split_train_test
import numpy as np

print "Loading data..."
mnist = fetch_mldata('MNIST original', data_home='/home/bdol/data')
train_data, test_data, train_target, test_target = split_train_test(mnist.data,
                                                                    mnist.target)
train_target = integral_to_indicator(train_target)
test_target = integral_to_indicator(test_target)
print "Done!"

np.seterr(all='ignore')
print "Training decision tree..."

# Comment the following two lines and uncomment the two lines following that
# if you want a faster version of the decision tree.
# dt = DecisionTree(6, 10)
# root = dt.train(train_data, train_target)
fast_dt = FastDecisionTree(10, 10)
root = fast_dt.train(train_data, train_target)
print "Done training!"
print "Testing..."
err = fast_dt.test(root, test_data, test_target)
print "Error rate: {0}".format(err)

print "Done!"

