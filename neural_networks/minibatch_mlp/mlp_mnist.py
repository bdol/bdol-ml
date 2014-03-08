import numpy as np
import sys
import bdolpyutils as bdp
import ConfigParser

class Weights:
  def __init__(self, sizes):
    self.W = []
    for i in range(0, len(sizes)):
      self.W.append(np.zeros(sizes[i]))

def rectified_linear(X):
  leqZeroIdx = X<=0
  Xprime = np.array(X)
  Xprime[leqZeroIdx] = 0
  return Xprime

def d_rectified_linear(X):
  leqZeroIdx = X<=0
  return leqZeroIdx*1

def linear(X):
  return X

#def d_linear(X):


def sigmoid(X):
  return 1/(1+np.exp(-X))

def d_sigmoid(X):
  return sigmoid(X)*(1-sigmoid(X))

def feed_forward(input, w, activation):
  output = activation(np.dot(input.T, w))
  return output

def forward_pass(input, W, activations, activationsD, doDropout):
  inp = np.atleast_2d(input)
  layer_output = np.append(inp, np.ones((inp.shape[0], 1)), 1)
  # Outputs at each layer
  z = [layer_output]
  # Evaluated derivatives of each layer
  d = []

  for i in range(0, len(W)):
    s_layer = np.dot(z[i], W[i])
    if doDropout:
      s_layer = np.random.binomial(1, 0.5, s_layer.shape)*s_layer

    z_layer = activations[i](s_layer)
    d_layer = activationsD[i](z_layer)

    # Add bias
    if i<len(W)-1:
      z_layer = np.append(z_layer, np.ones((z_layer.shape[0], 1)), 1)
    
    z.append(z_layer)
    d.append(d_layer)

  return z, d

def backwards_pass(z, d, y, W, activiationsD):
  # This is a vectorized implementation, still trying to figure out the matrix
  # representation
  for j in range(0, y.shape[0]):
    W_grad = []
    for i in range(0, len(W)):
      W_grad.append(np.zeros(W[i].shape))

    delta = []

    # Delta at the output layer
    e = (np.atleast_2d(z[-1][j, :]-y[j, :])).T
    delta.append(np.dot(np.diag(d[-1][j, :]), e))

    # Propagate error backwards
    for i in range(len(W)-1, 0, -1):
      delta.insert(0, np.dot(
                   np.dot(np.diag(d[i-1][j,:]), W[i][0:-1, :]),
                        delta[0]))

    # Compute the gradient for each weight
    for i in range(0, len(W)):
      W_grad[i] += np.dot(delta[i], np.atleast_2d(z[i][j, :])).T

  return W_grad, e

def test(X, Y, W, activations, activationsD):
  testErrs = 0
  for i in range(0, Y.shape[0]):
    z, d = forward_pass(X[i, :], W, activations, activationsD, False)
    yhat = np.argmax(z[-1])
    y = np.argmax(Y[i, :])
    if yhat != y:
      testErrs += 1

  return testErrs


digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#X_tr, Y_tr, X_v, Y_v, X_te, Y_te = bdp.loadMNIST('/Users/bdol/code/datasets/mnist/mnist.pkl', 
                                   #digits=digits, asBitVector=True)
X_tr, Y_tr, X_v, Y_v, X_te, Y_te = bdp.loadMNIST('/home/bdol/code/datasets/mnist/mnist.pkl', 
                                   digits=digits, asBitVector=True)
W = [np.random.randn(785, 50), np.random.randn(51, 50), np.random.randn(51, Y_tr.shape[1])]

activations = [sigmoid, sigmoid, sigmoid]
activationsD = [d_sigmoid, d_sigmoid, d_sigmoid]
doDropout = False

numEpochs = 2000
avg_error = 0
minibatchSize = 100

print "Training for "+str(numEpochs)+" epochs:"
for t in range(0, numEpochs):
  avg_error = 0
  for i in range(0, X_tr.shape[0], minibatchSize):
    z, d = forward_pass(X_tr[i:i+minibatchSize, :], W, activations, activationsD, doDropout)
    W_grad, e = backwards_pass(z, d, Y_tr[i:i+minibatchSize, :], W, activationsD)
    for j in range(0, len(W)):
      W[j] -= 0.002*W_grad[j]

    avg_error += e**2
    if i%100==0:
      bdp.progBar(i, X_tr.shape[0])

  testErrs = test(X_te, Y_te, W, activations, activationsD)

  print " Train: {0} Test: {1} Epoch: {2}".format(
        np.sum(avg_error)/(len(digits)*float(X_tr.shape[0])), testErrs, t)
  

print
print "Average error:", avg_error/X.shape[0]
