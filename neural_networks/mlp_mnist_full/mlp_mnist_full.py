import ConfigParser
import ast
import bdolpyutils as bdp
import numpy as np
import sys
import time

def rectified_linear(X):
  leqZeroIdx = X<=0
  Xprime = np.array(X)
  Xprime[leqZeroIdx] = 0
  return Xprime

def d_rectified_linear(X):
  gZeroIdx = X>0
  return gZeroIdx*1

def softmax(X):
  # TODO: there's gotta be a better way to do this without all the reshapes...
  # Use the log-sum-exp trick for numerical stability
  m = np.amax(X, axis=1)
  m = np.reshape(m, (len(m), -1))
  y_exp = np.exp(X-m)

  s = np.sum(y_exp, axis=1)
  s = np.reshape(s, (len(s), -1))
  try:
    r = y_exp/s
  except FloatingPointError:
    print y_exp
    sys.exit(1)
  return r

def d_softmax(X):
  # TODO: I don't think this is needed at the output layer, you should go
  # through the math
  return X

def sigmoid(X):
  r = 0
  try:
    r = 1/(1+np.exp(-X))
  except FloatingPointError:
    print X
    sys.exit(1)

  return r

def d_sigmoid(X):
  return sigmoid(X)*(1-sigmoid(X))

def linear(X):
  return X

class Layer:
  def __init__(self, size, activation, d_activation):
    # TODO: make 0.01 a sigma parameter
    self.W = 0.01*np.random.randn(size[0], size[1])
    self.activation = activation
    self.d_activation = d_activation
    self.z = np.zeros((size[0], size[1]))
    self.d = np.zeros((size[0], size[1]))
    self.a = 0

  def compute_activation(self, X, doDropout=False, dropoutProb=0.5):
    X_d = X
    # I think you should drop out columns here?
    if doDropout:
      X_d = X_d*np.random.binomial(1, (1-dropoutProb), X_d.shape)

    self.a = X_d.dot(self.W)
    self.z = self.activation(self.a)
    #self.d = self.d_activation(s)

    return self.z

class MLP:
  def __init__(self, layerSizes, activations, doDropout=False, droputProb=0.5,
      dropoutInputProb=0.2, wLenLimit=15):
    self.doDropout = doDropout
    self.dropoutProb = dropoutProb
    self.dropoutInputProb = dropoutInputProb

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
    self.previousGrad = []
    for i in range(0, len(layerSizes)-1):
      size = [layerSizes[i]+1, layerSizes[i+1]]
      activation = activationsMap[activations[i]]
      d_activation = d_activationsMap[activations[i]]

      l = Layer(size, activation, d_activation)
      self.layers.append(l)
      self.currentGrad.append(np.zeros(size))
      self.previousGrad.append(np.zeros(size))

  def forward_propagate(self, X):
    currentInput = np.atleast_2d(X)
    # First do forward propagation with the input data
    for i in range(0, len(self.layers)):
      # Append a 1 to the current input for the bias term
      currentInput = np.append(currentInput, np.ones((currentInput.shape[0], 1)), 1)
      # Now actually compute the activation
      # Input layer
      if i==0:
        currentInput = self.layers[i].compute_activation(currentInput,
            self.doDropout, self.dropoutInputProb)
      else:
        currentInput = self.layers[i].compute_activation(currentInput,
          self.doDropout, self.dropoutProb)
    return currentInput

  def backpropagate(self, output, X, Y, eta, momentum):
    # TODO: make the error function an input parameter. It's cross entropy in
    # the Hinton paper
    e = output-Y
    # First set up the gradients
    W_grad = []
    for i in range(0, len(self.layers)):
      W_grad.append( np.zeros(self.layers[i].W.shape) )

    # Backpropagate for each training example separately
    for j in range(0, Y.shape[1]):
      # delta_k's at the output layer
      deltas = [np.reshape(e[j, :], (len(e[j, :]), -1))]
      for i in range(len(self.layers)-2, -1, -1):
        W = self.layers[i+1].W
        n_units_j = self.layers[i].W.shape[1]
        n_units_k = self.layers[i+1].W.shape[1]
        deltas.insert(0, np.zeros((n_units_j, 1)))
        for jj in range(0, n_units_j):
          s = 0
          a_j = self.layers[i].a[j, jj]
          d = self.layers[i].d_activation(a_j)
          for kk in range(0, n_units_k):
            s += W[jj, kk]*deltas[1][kk]
          deltas[0][jj] = d*s

        ## Construct the matrix of derivatives D
        #D = np.diag(self.layers[i].d[j, :])
        ## We are at the output layer
        #if i==len(self.layers)-1:
          #deltas.append( e[j, :].T )
          #deltas[0] = np.reshape(deltas[0], (len(deltas[0]), -1))
        ## We are not at the output layer, so delta is scaled by the weights
        #if i<len(self.layers)-1:
          #W = self.layers[i+1].W[0:-1, :]
          #deltas.insert(0, D.dot(W).dot(deltas[0]))

      # Update the gradients
      for i in range(0, len(self.layers)):
        # We are at the input layer
        z_i = 0
        if i==0:
          z_i = X[j, :]
        else:
          z_i = self.layers[i-1].z[j, :]

        z_i = np.append(z_i, [1])
        z_i = np.reshape(z_i, (-1, len(z_i)))
        W_grad[i] += np.outer(deltas[i], z_i).T

    # Update the weights
    for i in range(0, len(self.layers)):
      self.currentGrad[i] = -eta*W_grad[i]
      #self.currentGrad[i] = momentum*self.previousGrad[i]-(1-momentum)*eta*W_grad[i]
      self.layers[i].W += self.currentGrad[i]
      self.previousGrad[i] = self.currentGrad[i]

      # Constrain the weights going to the hidden units if necessary
      if i<len(self.layers)-1:
        wLens = np.linalg.norm(self.layers[i].W, axis=0)**2
        wLenCorrections = np.ones([1, self.layers[i].W.shape[1]])
        wLenCorrections[0, np.where(wLens>wLenLimit)[0]] = wLens[wLens>wLenLimit]/wLenLimit
        self.layers[i].W = self.layers[i].W/(np.sqrt(wLenCorrections))

  # Propagate forward through the network, record the training error, train the
  # weights with backpropagation
  def train(self, X, Y, eta, momentum):
    output = self.forward_propagate(X)
    self.backpropagate(output, X, Y, eta, momentum)

  # Just pass the data forward through the network and return the predictions
  # for the given miniBatch
  def test(self, X):
    Yhat = np.zeros((X.shape[0], self.layers[-1].W.shape[1]))
    Yhat = self.forward_propagate(X)
    return Yhat

def RMSE(Y, Yhat):
  rmse = 0
  for i in range(0, Y.shape[0]):
    y_i = np.where(Y[i, :])[0]
    rmse += (1-Yhat[i, y_i])**2
  return np.sqrt(1/float(Y.shape[0])*rmse)[0]

def numErrs(Y, Yhat):
  Y_idx = np.argmax(Y, axis=1)
  Yhat_idx = np.argmax(Yhat, axis=1)
  return np.sum(Y_idx != Yhat_idx)

if __name__ == "__main__":
  # Load the parameters for this network from the initialization file
  cfg = ConfigParser.ConfigParser()
  cfg.read("params.ini")

  layerSizes = list(ast.literal_eval(cfg.get('net', 'layerSizes')))
  activations = cfg.get('net', 'activations').split(',')
  doDropout = ast.literal_eval(cfg.get('net', 'doDropout'))
  dropoutProb = ast.literal_eval(cfg.get('net', 'dropoutProb'))
  dropoutInputProb = ast.literal_eval(cfg.get('net', 'dropoutInputProb'))
  wLenLimit = ast.literal_eval(cfg.get('net', 'wLenLimit'))
  momentumInitial = ast.literal_eval(cfg.get('net', 'momentumInitial'))
  momentumFinal = ast.literal_eval(cfg.get('net', 'momentumFinal'))
  momentumT = ast.literal_eval(cfg.get('net', 'momentumT'))

  mlp = MLP(layerSizes, activations, doDropout, dropoutProb, dropoutInputProb,
      wLenLimit)

  # Additionally load the experiment parameters
  digits = list(ast.literal_eval(cfg.get('experiment', 'digits')))
  mnistPath = cfg.get('experiment', 'mnistPath')
  numEpochs = ast.literal_eval(cfg.get('experiment', 'numEpochs'))
  minibatchSize = ast.literal_eval(cfg.get('experiment', 'minibatchSize'))
  learningRate = ast.literal_eval(cfg.get('experiment', 'learningRate'))
  rateDecay = ast.literal_eval(cfg.get('experiment', 'rateDecay'))
  np.random.seed(1234)
  np.seterr(all='raise')
  np.seterr(under='ignore')

  # Set up the program options
  debugMode = ast.literal_eval(cfg.get('program', 'debugMode'))
  logToFile = ast.literal_eval(cfg.get('program', 'logToFile'))
  logFileBaseName = cfg.get('program', 'logFileBaseName')
  if logToFile:
    dateStr = time.strftime('_%Y-%m-%d_%H-%M')
    logFile = logFileBaseName+dateStr+".txt"
    f = open(logFile, "w")
    f.write('RMSE Train,Num. Errors Train,RMSE Valid.,Num. Errors Valid,RMSE Test,Num. Errors Test,learningRate,momentum,elapsedTime\n')

  # Load the corresponding data
  X_tr, Y_tr, X_v, Y_v, X_te, Y_te = bdp.loadMNIST(mnistPath, digits=digits,
      asBitVector=True)

  print "Training for "+str(numEpochs)+" epochs:"
  p = momentumInitial
  for t in range(0, numEpochs):
    startTime = time.time()

    if t< momentumT:
      p = 1/momentumT*momentumInitial + (1-1/momentumT)*momentumFinal
    else:
      p = momentumFinal

    for i in range(0, X_tr.shape[0], minibatchSize):
      mlp.train(X_tr[i:i+minibatchSize, :], Y_tr[i:i+minibatchSize],
          learningRate, p)

      if i%100==0:
        bdp.progBar(i, X_tr.shape[0])
    bdp.progBar(X_tr.shape[0], X_tr.shape[0])
    elapsedTime = (time.time()-startTime)
    print " Epoch {0}, learning rate: {1:.4f}, elapsed time: {2:.2f}s".format(t, learningRate, elapsedTime)

    learningRate = learningRate*rateDecay

    # Calculate training error
    YhatTrain = mlp.test(X_tr)
    rmseErrorTrain = RMSE(Y_tr, YhatTrain)
    numErrsTrain = numErrs(Y_tr, YhatTrain)
    YhatValid = mlp.test(X_v)
    rmseErrorValid = RMSE(Y_v, YhatValid)
    numErrsValid = numErrs(Y_v, YhatValid)
    YhatTest = mlp.test(X_te)
    rmseErrorTest = RMSE(Y_te, YhatTest)
    numErrsTest = numErrs(Y_te, YhatTest)

    errsStr = "Train RMSE: {0}\tTrain errors: {1}\n".format(rmseErrorTrain,
        numErrsTrain)
    errsStr += "Valid. RMSE: {0}\tValid. errors: {1}\n".format(rmseErrorValid,
        numErrsValid)
    errsStr += "Test RMSE: {0}\tTest errors: {1}\n".format(rmseErrorTest,
        numErrsTest)
    print errsStr

    logStr = "{0},{1},{2},{3},{4},{5},{6},{7},{8:.2f}\n".format(
                rmseErrorTrain, numErrsTrain,
                rmseErrorValid, numErrsValid,
                rmseErrorTest, numErrsTest,
                learningRate, p,
                elapsedTime)
    if logToFile:
      f.write(logStr)
    
  if logToFile:
    f.close()
