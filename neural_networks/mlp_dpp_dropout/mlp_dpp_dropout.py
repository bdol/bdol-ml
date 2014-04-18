import ConfigParser
import ast
import bdolpyutils as bdp
import numpy as np
import shutil
import sys
import time
import uuid

def rectified_linear(X):
  leqZeroIdx = X<=0
  Xprime = np.array(X)
  Xprime[leqZeroIdx] = 0
  return Xprime

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
  #TODO: this is wrong
  return X

def sigmoid(X):
  return 1.0/(1+np.exp(-X))

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
      X_d[:, 0:-1] = X_d[:, 0:-1]*np.random.binomial(1, (1-dropoutProb),
          (X_d.shape[0], X_d.shape[1]-1))

    self.a = X_d.dot(self.W)
    self.z = self.activation(self.a)
    self.d = self.d_activation(self.a)

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
    for i in range(0, len(layerSizes)-1):
      size = [layerSizes[i]+1, layerSizes[i+1]]
      activation = activationsMap[activations[i]]
      d_activation = d_activationsMap[activations[i]]

      l = Layer(size, activation, d_activation)
      self.layers.append(l)
      self.currentGrad.append(np.zeros(size))

  def forward_propagate(self, X):
    x_l = np.atleast_2d(X)
    for i in range(0, len(self.layers)):
      x_l = np.append(x_l, np.ones((x_l.shape[0], 1)), 1)
      if i==0: # We're at the input layer
        x_l = self.layers[i].compute_activation(x_l, doDropout, dropoutInputProb)
      else:
        x_l = self.layers[i].compute_activation(x_l, doDropout, dropoutProb)

    return x_l

  def xent_cost(self, X, Y, Yhat):
    E = 0
    for i in range(0, Y.shape[0]):
      y = np.argmax(Y[i, :])
      E -= np.log(Yhat[i, y])
    return E

  def check_gradient(self, X, Y, eta, momentum):
    eps = 1E-4
    output = self.forward_propagate(X)
    W_grad = self.calculate_gradient(output, X, Y, eta, momentum)
    
    W_initial = []
    for i in range(0, len(self.layers)):
      W_initial.append(np.copy(self.layers[i].W))

    for i in range(0, len(self.layers)):
      W = self.layers[i].W
      for j in range(0, W.shape[0]):
        for k in range(0, W.shape[1]):
          self.layers[i].W[j,k] += eps
          out_p = self.forward_propagate(X)
          E_p = self.xent_cost(X, Y, out_p)
          self.layers[i].W[j,k] = W_initial[i][j,k]
          self.layers[i].W[j,k] -= eps 
          out_m = self.forward_propagate(X)
          E_m = self.xent_cost(X, Y, out_m)
          self.layers[i].W[j,k] = W_initial[i][j,k]

          g_approx = (E_p-E_m)/(2*eps)
          if abs(g_approx-W_grad[i][j,k])>1E-4:
            print "Gradient checking failed for ",i,j,k,abs(g_approx-W_grad[i][j,k])

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
      if i==0:
        z_i = X
      else:
        z_i = self.layers[i-1].z
      z_i = np.append(z_i, np.ones((z_i.shape[0], 1)), 1)
      W_grad[i] = (deltas[i].dot(z_i)).T

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
  #cfg.set('net', 'momentumT', 'fuck')
  #with open('params_test.ini', 'wb') as newconfig:
    #cfg.write(newconfig)
  #sys.exit(0)

  mlp = MLP(layerSizes, activations, doDropout, dropoutProb, dropoutInputProb,
      wLenLimit)

  # Additionally load the experiment parameters
  digits = list(ast.literal_eval(cfg.get('experiment', 'digits')))
  mnistPath = cfg.get('experiment', 'mnistPath')
  numEpochs = ast.literal_eval(cfg.get('experiment', 'numEpochs'))
  minibatchSize = ast.literal_eval(cfg.get('experiment', 'minibatchSize'))
  learningRate = ast.literal_eval(cfg.get('experiment', 'learningRate'))
  rateDecay = ast.literal_eval(cfg.get('experiment', 'rateDecay'))
  checkGradient = ast.literal_eval(cfg.get('experiment', 'checkGradient'))
  np.random.seed(1234)
  # This option will continue the experiment for a certain number of epochs
  # after we have gotten 0 training errors
  numEpochsAfterOverfit = ast.literal_eval(cfg.get('experiment',
    'numEpochsAfterOverfit'))
  numEpochsRemaining = numEpochsAfterOverfit

  # Set up the program options
  debugMode = ast.literal_eval(cfg.get('program', 'debugMode'))
  logToFile = ast.literal_eval(cfg.get('program', 'logToFile'))
  logFileBaseName = cfg.get('program', 'logFileBaseName')
  if logToFile:
    dateStr = time.strftime('%Y-%m-%d_%H-%M')
    # Add a UUID so we can track this experiment
    uuidStr = uuid.uuid4()
    logFile = logFileBaseName+"_"+dateStr+"_"+str(uuidStr)+".txt"
    f = open(logFile, "w")
    f.write('Num. Errors Train,Num. Errors Test,learningRate,momentum,elapsedTime\n')
    # Also copy the params over for posterity
    paramsCopyStr = logFileBaseName+"_params_"+str(uuidStr)+".ini"
    shutil.copyfile("params.ini", paramsCopyStr)

  # Load the corresponding data
  X_tr, Y_tr, X_te, Y_te = bdp.loadMNISTnp(mnistPath, digits=digits,
      asBitVector=True)

  if checkGradient:
    print "Checking gradient..."
    mlp.check_gradient(X_tr[0:10, :], Y_tr[0:10, :], learningRate, 0)
    print "Gradient checking complete."

  print "Training for "+str(numEpochs)+" epochs:"
  p = momentumInitial
  for t in range(0, numEpochs):
    startTime = time.time()

    for i in range(0, X_tr.shape[0], minibatchSize):
      mlp.train(X_tr[i:i+minibatchSize, :], Y_tr[i:i+minibatchSize],
          learningRate, p)


      if i%(10*minibatchSize)==0:
        bdp.progBar(i, X_tr.shape[0])
    bdp.progBar(X_tr.shape[0], X_tr.shape[0])

    elapsedTime = (time.time()-startTime)
    print " Epoch {0}, learning rate: {1:.4f}, momentum: {2:.4f} elapsed time: {3:.2f}s".format(t, learningRate, p, elapsedTime)

    # Decay the learning rate
    learningRate = learningRate*rateDecay

    # Update the momentum
    if t < momentumT:
      p = (1.0-float(t)/momentumT)*momentumInitial + (float(t)/momentumT)*momentumFinal
    else:
      p = momentumFinal

    # Calculate errors
    YhatTrain = mlp.test(X_tr)
    numErrsTrain = numErrs(Y_tr, YhatTrain)
    YhatTest = mlp.test(X_te)
    numErrsTest = numErrs(Y_te, YhatTest)

    errsStr = "Train errors: {0}\n".format(numErrsTrain)
    errsStr += "Test errors: {0}\n".format(numErrsTest)
    print errsStr

    logStr = "{0},{1},{2},{3},{4:.2f}\n".format(
                numErrsTrain, numErrsTest, learningRate, p, elapsedTime)
    if logToFile:
      f.write(logStr)

    if numErrsTrain == 0 and numEpochsAfterOverfit > 0:
      print "No training errors. Continuing for {0} more epochs.".format(numEpochsRemaining) 
      numEpochsAfterOverfit -= 1
    elif numErrsTrain == 0 and numEpochsRemaining == 0:
      print "No training errors. Stopping."
      break
    elif numErrsTrain > 0:
      numEpochsRemaining = numEpochsAfterOverfit

  if logToFile:
    f.close()
