"""
Here we show an example of using the greedy algorithm to perform feature
selection for Naive Bayes. We use two lists from the 20newsgroups dataset and
select k << D features to use for a NB classifier.

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

from sklearn.datasets import fetch_20newsgroups_vectorized, fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from submodularity.utils.greedy_max import greedy_submodular_max
from submodularity.utils.submodular_functions import information_gain_nb
import numpy as np

from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

from nb_selection_functions import nb_feature_selection, unfactorized_feature_selection

print "(Down)loading data..."
newsgroups_train = fetch_20newsgroups(subset='train',
                                      data_home='/home/bdol/data',
                                      categories=['alt.atheism',
                                                  'talk.religion.misc'])
newsgroups_test = fetch_20newsgroups(subset='test',
                                      data_home='/home/bdol/data',
                                      categories=['alt.atheism',
                                                  'talk.religion.misc'])

print "Done!"

print "Vectorizing features..."
vectorizer = CountVectorizer(binary=True)
print "Done!"

train_data = vectorizer.fit_transform(newsgroups_train.data)
train_target = newsgroups_train.target

A = nb_feature_selection(train_data, train_target, 8)
print "Factored:",[vectorizer.get_feature_names()[a] for a in A]

A = unfactorized_feature_selection(train_data, train_target, 8)
print "Non-factored:",[vectorizer.get_feature_names()[a] for a in A]
