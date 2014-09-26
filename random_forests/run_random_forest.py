"""
An example that runs my implementation of random forests.

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

from random_forest import RandomForest
from sklearn.datasets import fetch_mldata
from data_utils import integral_to_indicator, split_train_test
import numpy as np

print "Loading data..."
mnist = fetch_mldata('MNIST original', data_home='/home/bdol/data')
train_data, test_data, train_target, test_target = split_train_test(mnist.data,
                                                                    mnist.target)
train_target = integral_to_indicator(train_target)
test_target_integral = integral_to_indicator(test_target)
print "Done!"

np.seterr(all='ignore')

print "Training random forest..."
rf = RandomForest(20, 10, 3, boot_percent=0.3, feat_percent=0.1, debug=True)
rf.train(train_data, train_target)
print "Done training!"

print "Testing..."
yhat = rf.test(test_data, test_target_integral)
err = (np.sum(yhat != test_target[:, None]).astype(float))/test_target.shape[0]
print "Error rate: {0}".format(err)

print "Done!"
