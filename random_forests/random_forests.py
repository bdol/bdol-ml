import bdolpyutils as bdp
import numpy as np
import sys
from scipy import stats

class Node():
  def __init__(self):
    self.left = None
    self.right = None
    self.terminal = False
    self.fidx = []
    self.fval = []
    self.value = []

def plogp(x):
  e = x*np.log2(x)
  # Set values outside the range of log to 0
  e[np.isinf(e)] = 0
  e[np.isnan(e)] = 0
  return e

def entropy(x):
  return -np.sum(plogp(x), axis=0)

def choose_feature(X, Y, Xrange, colidx, m):
  py = np.mean(Y, axis=0)
  H = entropy(py)

  IG = np.zeros((len(Xrange), 1))
  splitVals = np.zeros((len(Xrange), 1))

  fval = 0.004
  split = (X <= fval).astype(float)
  px = np.mean(split, axis=0)

  sum_x = np.sum(split, axis=0).astype(float)
  sum_notx = X.shape[0] - sum_x

  py_given_x = np.zeros((Y.shape[1], X.shape[1]))
  py_given_notx = np.zeros((Y.shape[1], X.shape[1]))
  for k in range(0, Y.shape[1]):
    y_given_x = ((Y[:, k]==1)[:, None] & (split==1)).astype(float)
    y_given_notx = ((Y[:, k]==1)[:, None] & (split==0)).astype(float)

    py_given_x[k, :] = np.sum(y_given_x, axis=0)/sum_x
    py_given_notx[k, :] = np.sum(y_given_notx, axis=0)/sum_notx

  # Compute the conditional entropy and information gain
  cond_H = px*entropy(py_given_x) + (1-px)*entropy(py_given_notx)
  ig = H-cond_H

  # Select m random features from the remaining columns
  featureIdx = np.random.choice(colidx, m)

  # Select the feature that gives the most informative split
  max_ig = np.max(ig[featureIdx])
  fidx = np.argmax(ig[featureIdx])

  # Make sure that we use the original index and not the index of the truncated
  # data matrix
  fidx = featureIdx[fidx]

  return fidx, fval, max_ig

def split_node(X, Y, Xrange, defaultValue, colidx, depth, depthLimit):
  py = np.mean(Y, axis=0)

  node = Node()

  if depth==depthLimit or len(colidx)==0 or np.max(py)==1 or Y.shape[0]<=1:
    node.terminal = True
    if Y.shape[0] == 0:
      node.value = defaultValue
    else:
      node.value = np.mean(Y, 0)

    print "*** depth: {0} [{1}]: Leaf predictions: {2}".format(
        depth, Y.shape[0], node.value)
    return node

  node.fidx, node.fval, max_ig = choose_feature(X, Y, Xrange, colidx, m)
  node.value = np.mean(Y, 0)
  colidx = np.delete(colidx, np.where(colidx==node.fidx))
  leftidx = X[:, node.fidx] <= node.fval
  rightidx = X[:, node.fidx] > node.fval
  
  print "depth: {0} [{1}]: Split on feature {2}. L/R = {3}/{4}".format(
      depth, Y.shape[0], node.fidx, np.sum(leftidx), np.sum(rightidx))

  node.left = split_node(X[leftidx, :], Y[leftidx], Xrange, node.value, colidx, depth+1, depthLimit)
  node.right = split_node(X[rightidx, :], Y[rightidx], Xrange, node.value, colidx, depth+1, depthLimit)

  return node

def trainTree(X, Y, m, depthLimit):
  # Compute the range of values for each feature
  Xrange = []
  for j in range(0, X.shape[1]):
    Xrange.append(np.unique(X[:, j]))

  return split_node(X, Y, Xrange, np.mean(Y.astype(float), axis=0), np.arange(0, X.shape[1]-1), 0, depthLimit)

def dt_value(root, x):
  node = root
  while not node.terminal:
    if x[node.fidx]<=node.fval:
      node = node.left
    else:
      node = node.right
  return node.value

def testTree(root, X, Y):
  errs = 0.0
  for i in range(0, X.shape[0]):
    yhat = np.argmax(dt_value(root, X[i, :]))
    if Y[i, yhat] != 1:
      errs += 1.0

  return errs/X.shape[0]

def testForest(roots, X, Y):
  errs = 0.0

  for i in range(0, X.shape[0]):
    votes = []
    for r in roots:
      yhat = np.argmax(dt_value(r, X[i, :]))
      votes.append(yhat)
    yhatEnsemble = stats.mode(votes)
    if Y[i, int(yhatEnsemble[0])] != 1:
      errs += 1.0

  return errs/X.shape[0]

# Turn off runtime warnings for invalid or divide errors
# These arise when calculating the entropy. We set any invalid entropy
# calculations (e.g. log(0)) to 0.
np.seterr(all='ignore')

# When printing the terminal values, only show to 2 decimals of precision
np.set_printoptions(precision=2, suppress=True)

X_tr, Y_tr, X_va, Y_va, X_te, Y_te = \
    bdp.loadMNIST('/home/bdol/code/datasets/mnist/mnist.pkl', asBitVector=True)

numSplits = 10

numTrees = 101
depth = 6
m = np.sqrt(X_tr.shape[1])

Nsample = np.floor(2.0/3.0*X_tr.shape[0])
roots = []

for i in range(0, numTrees):
  print "\n\n=============== Training tree {0} ===============".format(i+1)

  idx = np.random.choice(X_tr.shape[0], Nsample)
  X_t = X_tr[idx, :]
  Y_t = Y_tr[idx, :]

  root = trainTree(X_t, Y_t, m, depth)
  roots.append(root)

  print "Evaluating forest..."
  single_err = testTree(root, X_tr, Y_tr)
  tr_errs = testForest(roots, X_tr, Y_tr)
  print "Single tree error: {0}\nTraining error on current forest ({1} tree(s)): {2}".format(single_err, i+1, tr_errs)

print
print "=============== Testing ==============="
va_errs = testForest(roots, X_va, Y_va)
te_errs = testForest(roots, X_te, Y_te)
print "Average error on validation set: {0}".format(va_errs)
print "Average error on test set: {0}".format(te_errs)
