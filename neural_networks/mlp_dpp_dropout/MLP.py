import numpy as np
import dpp

def rectified_linear(X):
  return np.maximum(X, 0.0)

def d_rectified_linear(X):
  gZeroIdx = X>0
  return gZeroIdx*1

def softmax(X):
  # Use the log-sum-exp trick for numerical stability
  m = np.atleast_2d(np.amax(X, axis=1)).T
  y_exp = np.exp(X-m)

  s = np.atleast_2d(np.sum(y_exp, axis=1)).T

  return y_exp/s

def d_softmax(X):
  return X

def sigmoid(X):
  return 1.0/(1+np.exp(-X))

def d_sigmoid(X):
  return sigmoid(X)*(1-sigmoid(X))

def linear(X):
  return X


class Layer:
  def __init__(self, size, activation, d_activation):
    self.W = 0.01*np.random.randn(size[0], size[1])

    self.prev_Z = None
    self.activation = activation
    self.d_activation = d_activation
    self.z = np.zeros((size[0], size[1]))
    self.d = np.zeros((size[0], size[1]))
    self.a = 0

  def random_dropout(self, dropoutProb):
    if dropoutProb == 0:
      return

    d_idx = np.random.binomial(1, (1-dropoutProb), self.prevZ.shape[1]-1)
    self.prevZ[:, 0:-1] = self.prevZ[:, 0:-1]*d_idx

  def dpp_dropin_norm(self, dropoutProb):
    if dropoutProb == 0:
      return

    W_n = self.W[0:-1, :]/(np.atleast_2d(np.linalg.norm(self.W[0:-1, :], axis=1)).T)
    L = (W_n.dot(W_n.T))**2
    D, V = dpp.decompose_kernel(L)
    
    k = int(np.floor((1-dropoutProb)*self.W.shape[0]))
    J = dpp.sample_k(k, D, V)
    d_idx = np.zeros((self.W.shape[0]-1, 1))
    d_idx[J.astype(int)] = 1
  
    self.prevZ[:, 0:-1] = self.prevZ[:, 0:-1]*d_idx.T

  def dpp_dropin(self, dropoutProb):
    if dropoutProb == 0:
      return

    W_n = self.W[0:-1, :]
    L = (W_n.dot(W_n.T))**2
    D, V = dpp.decompose_kernel(L)
    
    k = int(np.floor((1-dropoutProb)*self.W.shape[0]))
    J = dpp.sample_k(k, D, V)
    d_idx = np.zeros((self.W.shape[0]-1, 1))
    d_idx[J.astype(int)] = 1
  
    self.prevZ[:, 0:-1] = self.prevZ[:, 0:-1]*d_idx.T

  def dpp_dropin_EY(self, dropoutProb):
    if dropoutProb == 0:
      return

    p = (1-dropoutProb)
    W_n = self.W[0:-1, :]
    L = (W_n.dot(W_n.T))**2
    D, V = dpp.decompose_kernel(L)
    (kmax, d) = dpp.analyze_bDPP(D)
    print kmax
    if kmax>=(p*L.shape[0]):
      print "DPP!"
      J = dpp.sample_EY(D, V, p*L.shape[0])
      d_idx = np.zeros((self.W.shape[0]-1, 1))
      d_idx[J.astype(int)] = 1
      self.prevZ[:, 0:-1] = self.prevZ[:, 0:-1]*d_idx.T
      
    else:
      self.random_dropout(dropoutProb)
    
    #print dpp.analyze_bDPP(D)

  def dpp_dropout_norm(self, dropoutProb):
    if dropoutProb == 0:
      return

    W_n = self.W[0:-1, :]/(np.atleast_2d(np.linalg.norm(self.W[0:-1, :], axis=1)).T)
    L = (W_n.dot(W_n.T))**2
    D, V = dpp.decompose_kernel(L)
    
    k = int(np.floor((1-dropoutProb)*self.W.shape[0]))
    J = dpp.sample_k(k, D, V)
    d_idx = np.ones((self.W.shape[0]-1, 1))
    d_idx[J.astype(int)] = 0
  
    self.prevZ[:, 0:-1] = self.prevZ[:, 0:-1]*d_idx.T

  def dpp_dropout(self, dropoutProb):
    if dropoutProb == 0:
      return

    W_n = self.W[0:-1, :]
    L = (W_n.dot(W_n.T))**2
    D, V = dpp.decompose_kernel(L)
    
    k = int(np.floor((1-dropoutProb)*self.W.shape[0]))
    J = dpp.sample_k(k, D, V)
    d_idx = np.ones((self.W.shape[0]-1, 1))
    d_idx[J.astype(int)] = 0
  
    self.prevZ[:, 0:-1] = self.prevZ[:, 0:-1]*d_idx.T

  def dpp_dropin_uniform(self, dropoutProb):
    if dropoutProb == 0:
      return

    p = (1-dropoutProb)
    L = (p/(1-p))*np.eye(self.W.shape[0]-1)
    D, V = dpp.decompose_kernel(L)
    J = dpp.sample(D, V)
    d_idx = np.zeros((self.W.shape[0]-1, 1))
    d_idx[J.astype(int)] = 1
  
    self.prevZ[:, 0:-1] = self.prevZ[:, 0:-1]*d_idx.T
  def compute_activation(self, X, doDropout=False, dropoutProb=0.5,
      testing=False, dropoutSeed=None):
    self.prevZ = np.copy(X)
    # I think you should drop out columns here?
    if doDropout:
      # We are not testing, so do dropout
      if not testing:
	self.dropoutFunction(dropoutProb)
        self.a = self.prevZ.dot(self.W)

      # We are testing, so we don't do dropout but we do scale the weights
      if testing:
        self.a = self.prevZ[:, 0:-1].dot(self.W[0:-1, :]*(1-dropoutProb))
        self.a += np.outer(self.prevZ[:, -1], self.W[-1, :])
    else:
      self.a = self.prevZ.dot(self.W)

    self.z = self.activation(self.a)
    self.d = self.d_activation(self.a)

    return self.z

class MLP:
  def __init__(self, layerSizes, activations, doDropout=False,
      dropoutType='nodropout', dropoutProb=0.5,
      dropoutInputProb=0.2, wLenLimit=15):
    self.doDropout = doDropout
    self.dropoutProb = dropoutProb
    self.dropoutInputProb = dropoutInputProb
    self.wLenLimit = wLenLimit

    # Activations map - we need this to map from the strings in the *.ini file
    # to actual function names
    activationsMap = {'sigmoid': sigmoid,
                      'rectified_linear': rectified_linear,
                      'softmax': softmax}
    d_activationsMap = {'sigmoid': d_sigmoid,
                        'rectified_linear': d_rectified_linear,
                        'softmax': d_softmax}
    # Initialize each layer with the given parameters
    self.layers = []
    self.currentGrad = []
    for i in range(0, len(layerSizes)-1):
      size = [layerSizes[i]+1, layerSizes[i+1]]
      activation = activationsMap[activations[i]]
      d_activation = d_activationsMap[activations[i]]

      l = Layer(size, activation, d_activation)
      dropoutTypeMap = {'nodropout': None,
                        'dpp_dropout_norm': l.dpp_dropout_norm,
                        'dpp_dropout': l.dpp_dropout,
                        'dpp_dropin_norm': l.dpp_dropin_norm,
                        'dpp_dropin': l.dpp_dropin,
                        'dpp_dropin_uniform': l.dpp_dropin_uniform,
                        'dpp_dropin_EY': l.dpp_dropin_EY,
                        'random': l.random_dropout}

      l.dropoutFunction = dropoutTypeMap[dropoutType]
      self.layers.append(l)
      self.currentGrad.append(np.zeros(size))

  def forward_propagate(self, X, testing=False, dropoutSeeds=None):
    x_l = np.atleast_2d(X)
    for i in range(0, len(self.layers)):
      x_l = np.append(x_l, np.ones((x_l.shape[0], 1)), 1)
      if i==0: # We're at the input layer
        if dropoutSeeds:
          x_l = self.layers[i].compute_activation(x_l, self.doDropout,
              self.dropoutInputProb, testing, dropoutSeeds[i])
        else:
          x_l = self.layers[i].compute_activation(x_l, self.doDropout,
              self.dropoutInputProb, testing)
      else:
        if dropoutSeeds:
          x_l = self.layers[i].compute_activation(x_l, self.doDropout, self.dropoutProb,
              testing, dropoutSeeds[i])
        else:
          x_l = self.layers[i].compute_activation(x_l, self.doDropout, self.dropoutProb,
              testing)

    return x_l

  def xent_cost(self, X, Y, Yhat):
    E = np.array([0]).astype(np.float64)
    for i in range(0, Y.shape[0]):
      y = np.argmax(Y[i, :])
      E -= np.log(Yhat[i, y]).astype(np.float64)
      
    return E

  def check_gradient(self, X, Y, eta, momentum):
    eps = 1E-4
    dropoutSeeds = [232, 69, 75, 333]
    output = self.forward_propagate(X, dropoutSeeds=dropoutSeeds)
    W_grad = self.calculate_gradient(output, X, Y, eta, momentum)

    for i in range(0, len(self.layers)):
      np.savetxt("W_grad_"+str(i), W_grad[i])
      np.savetxt("W_"+str(i), self.layers[i].W)
    
    W_initial = []
    for i in range(0, len(self.layers)):
      W_initial.append(np.copy(self.layers[i].W))

    for i in range(0, len(self.layers)):
      W = self.layers[i].W
      print " Checking layer",i
      layer_err = 0
      for j in range(0, W.shape[0]):
        for k in range(0, W.shape[1]):
          self.layers[i].W[j,k] += eps
          out_p = self.forward_propagate(X, dropoutSeeds=dropoutSeeds)
          E_p = self.xent_cost(X, Y, out_p)
          self.layers[i].W[j,k] = W_initial[i][j,k]
          self.layers[i].W[j,k] -= eps 
          out_m = self.forward_propagate(X, dropoutSeeds=dropoutSeeds)
          E_m = self.xent_cost(X, Y, out_m)
          self.layers[i].W[j,k] = W_initial[i][j,k]

          g_approx = (E_p-E_m)/(2*eps)
          g_calc = W_grad[i][j,k]
          err = abs(g_approx-g_calc)/(abs(g_approx)+abs(g_calc)+1E-10)
          layer_err += err
          if err>1E-3:
          #if g_approx == 0 and g_calc != 0:
            print " Gradient checking failed for ",i,j,k,g_approx,W_grad[i][j,k],E_p, E_m, err

        bdp.progBar(j, self.layers[i].W.shape[0])
      print layer_err

  def calculate_gradient(self, output, X, Y, eta, momentum):
    # First set up the gradients
    W_grad = []
    for i in range(0, len(self.layers)):
      W_grad.append( np.zeros(self.layers[i].W.shape) )

    e = output-Y

    # Backpropagate for each training example separately
    deltas = [e.T]
    for i in range(len(self.layers)-2, -1, -1):
      W = self.layers[i+1].W[0:-1, :]
      deltas.insert(0, np.multiply(self.layers[i].d.T, W.dot(deltas[0])))

    for i in range(0, len(self.layers)):
      W_grad[i] = (deltas[i].dot(self.layers[i].prevZ)).T

    return W_grad

  def backpropagate(self, output, X, Y, eta, momentum):
    W_grad = self.calculate_gradient(output, X, Y, eta, momentum)

    # Update the current gradient, and step in that direction
    for i in range(0, len(self.layers)):
      self.currentGrad[i] = momentum*self.currentGrad[i] - (1.0-momentum)*eta*W_grad[i]
      self.layers[i].W += self.currentGrad[i]
      #self.previousGrad[i] = np.copy(self.currentGrad[i])

      # Constrain the weights going to the hidden units if necessary
      wLens = np.linalg.norm(self.layers[i].W, axis=0)**2
      wLenCorrections = np.ones([1, self.layers[i].W.shape[1]])
      wLenCorrections[0, np.where(wLens>self.wLenLimit)[0]] = wLens[wLens>self.wLenLimit]/self.wLenLimit
      self.layers[i].W = self.layers[i].W/(np.sqrt(wLenCorrections))

  # Propagate forward through the network, record the training error, train the
  # weights with backpropagation
  def train(self, X, Y, eta, momentum):
    output = self.forward_propagate(X, testing=False)
    self.backpropagate(output, X, Y, eta, momentum)

  # Just pass the data forward through the network and return the predictions
  # for the given miniBatch
  def test(self, X):
    Yhat = np.zeros((X.shape[0], self.layers[-1].W.shape[1]))
    Yhat = self.forward_propagate(X, testing=True)
    return Yhat


