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
  # Use the log-sum-exp trick for numerical stability
  m = np.amax(X, axis=1)
  m = np.reshape(m, (len(m), -1))
  y_exp = np.exp(X-m)

  s = np.sum(y_exp, axis=1)
  s = np.reshape(s, (len(s), -1))
  #try:
    #r = y_exp/s
  #except FloatingPointError:
    #print y_exp
    #print "softmax err"
    #sys.exit(1)
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
    X_d = np.copy(X)
    # I think you should drop out columns here?
    if doDropout:
      X_d = X_d*np.random.binomial(1, (1-dropoutProb), X_d.shape)

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
    self.defcon5 = False

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
    x_l = np.atleast_2d(X)
    for i in range(0, len(self.layers)):
      x_l = np.append(x_l, np.ones((x_l.shape[0], 1)), 1)
      x_l = self.layers[i].compute_activation(x_l)

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
          if abs(g_approx-W_grad[i][j,k])>1E-8:
            print "Gradient checking failed for ",i,j,k,abs(g_approx-W_grad[i][j,k])

  def calculate_gradient(self, output, X, Y, eta, momentum):
    # First set up the gradients
    W_grad = []
    deltas = []
    for i in range(0, len(self.layers)):
      W_grad.append( np.zeros(self.layers[i].W.shape) )
      deltas.append( np.zeros((self.layers[i].W.shape[1], 1)))

    e = output-Y

    # Backpropagate for each training example separately
    for j in range(0, Y.shape[0]):
      # Delta at the output layer
      deltas[-1] = e[j, :].T
      # Backpropagation to determine the deltas at each layer
      for i in range(len(self.layers)-2, -1, -1):
        D = np.diag(self.layers[i].d[j, :])
        W = self.layers[i+1].W[0:-1, :]
        deltas[i] = D.dot(W.dot(deltas[i+1]))

        # Add the gradient for this example
        #z_i = 0
        ## Gradient at the input layer
        #if i==0:
          #z_i = X[j, :]
        #else:
          #z_i = self.layers[i-1].z[j, :]
        #z_i = np.append(z_i, [1])
        #W_grad[i] += np.outer(z_i, deltas[i])

      for i in range(0, len(self.layers)):
        z_i = 0
        # Gradient at the input layer
        if i==0:
          z_i = X[j, :]
        else:
          z_i = self.layers[i-1].z[j, :]
        z_i = np.append(z_i, [1])
        W_grad[i] += np.outer(z_i, deltas[i])

      if self.defcon5 and j==4:
        for i in range(0, len(self.layers)):
          np.savetxt("Wdfc_"+str(i), self.layers[i].W)
          np.savetxt("Ddfc_"+str(i), self.layers[i].d)
          np.savetxt("delta_"+str(i), deltas[i])
        np.savetxt("XXdfc", X)
        np.savetxt("YYdfc", Y)
        np.savetxt("outputdfc", output)
        np.savetxt("e", e)
        self.defcon5=False

    return W_grad

  def backpropagate(self, output, X, Y, eta, momentum):
    W_grad = self.calculate_gradient(output, X, Y, eta, momentum)

    # Update the current gradient, and step in that direction
    for i in range(0, len(self.layers)):
      self.currentGrad[i] = momentum*self.previousGrad[i] - (1.0-momentum)*eta*W_grad[i]
      self.layers[i].W += self.currentGrad[i]
      self.previousGrad[i] = np.copy(self.currentGrad[i])

      # Constrain the weights going to the hidden units if necessary
      #if i<len(self.layers)-1:
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
  checkGradient = ast.literal_eval(cfg.get('experiment', 'checkGradient'))
  np.random.seed(1234)

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
  X_tr, Y_tr, X_te, Y_te = bdp.loadMNISTnp(mnistPath, digits=digits,
      asBitVector=True)

  if checkGradient:
    print "Checking gradient..."
    mlp.check_gradient(X_tr[0:10, :], Y_tr[0:10, :], learningRate, 0)
    print "Gradient is correct!"

  print "Training for "+str(numEpochs)+" epochs:"
  p = momentumInitial
  for t in range(0, numEpochs):
    startTime = time.time()

    for i in range(0, X_tr.shape[0], minibatchSize):
      mlp.train(X_tr[i:i+minibatchSize, :], Y_tr[i:i+minibatchSize],
          learningRate, p)


      if i%minibatchSize==0:
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
    np.savetxt('mnistyhat.txt', YhatTrain)
    np.savetxt('mnisty.txt', Y_tr)
    rmseErrorTrain = RMSE(Y_tr, YhatTrain)
    numErrsTrain = numErrs(Y_tr, YhatTrain)
    YhatTest = mlp.test(X_te)
    rmseErrorTest = RMSE(Y_te, YhatTest)
    numErrsTest = numErrs(Y_te, YhatTest)

    errsStr = "Train RMSE: {0}\tTrain errors: {1}\n".format(rmseErrorTrain,
        numErrsTrain)
    errsStr += "Test RMSE: {0}\tTest errors: {1}\n".format(rmseErrorTest,
        numErrsTest)
    print errsStr

    logStr = "{0},{1},{2},{3},{4},{5},{6:.2f}\n".format(
                rmseErrorTrain, numErrsTrain,
                rmseErrorTest, numErrsTest,
                learningRate, p,
                elapsedTime)
    if logToFile:
      f.write(logStr)

    #if numErrsTest > 1000:
      #for i in range(0, len(mlp.layers)):
        #np.savetxt("W_"+str(i), mlp.layers[i].W)
      #np.savetxt("XX", X_te[0:100, :])
      #np.savetxt("YY", Y_te[0:100, :])
      #print "Coadaptation detected..."
      #mlp.defcon5 = True
      ##if logToFile:
        ##f.close()
      
    
  if logToFile:
    f.close()
