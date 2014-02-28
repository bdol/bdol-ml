import numpy as np
import sys

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

def sigmoid(X):
  return 1/(1+np.exp(-X))

def d_sigmoid(X):
  return sigmoid(X)*(1-sigmoid(X))

def feed_forward(input, w, activation):
  output = activation(np.dot(input.T, w))
  return output

def forward_pass(input, W, activations, activationsD, doDropout):
  layer_output = np.append(input, 1)
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
      z_layer = np.append(z_layer, 1)
    
    z.append(z_layer)
    d.append(d_layer)

  return z, d

def backwards_pass(z, d, y, W, activiationsD):
  W_grad = []
  delta = []

  e = z[-1]-y
  # Delta at the output layer
  delta.append(np.diag(d[-1])*e)
  # Propagate error backwards
  for i in range(len(W)-1, 0, -1):
    delta.insert(0, np.dot(
                      np.dot(np.diag(d[i-1]), W[i][0:-1, :]),
                      delta[0]))

  # Compute the gradient for each weight
  for i in range(0, len(W)):
    W_grad.append(np.outer(delta[i], z[i]).T)

  return W_grad, e

# Single layer network
X = np.random.rand(100, 2)
y = (np.sqrt(np.sum(X**2, 1))>0.7)*1
W = [np.random.randn(3, 10), np.random.randn(11, 10), np.random.randn(11, 1)]
activations = [sigmoid, sigmoid, sigmoid]
activationsD = [d_sigmoid, d_sigmoid, sigmoid]
doDropout = False

numEpochs = 5000
avg_error = 0
for t in range(0, numEpochs):
  avg_error = 0
  for i in range(0, X.shape[0]):
    z, d = forward_pass(X[i, :], W, activations, activationsD, doDropout)
    W_grad, e = backwards_pass(z, d, y[i], W, activationsD)
    for i in range(0, len(W)):
      W[i] -= 0.002*W_grad[i]

    avg_error += e**2

  if t%10==0:
    print avg_error/float(X.shape[0])

