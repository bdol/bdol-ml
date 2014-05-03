import ConfigParser
import ast
import bdolpyutils as bdp
import MLP
import numpy as np
import shutil
import sys
import time
import uuid

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
  cfg.read(sys.argv[1])

  layerSizes = list(ast.literal_eval(cfg.get('net', 'layerSizes')))
  activations = cfg.get('net', 'activations').split(',')
  dropoutType = cfg.get('net', 'dropoutType')
  if dropoutType == 'nodropout':
    doDropout = False
  else:
    doDropout = True
  dropoutProb = ast.literal_eval(cfg.get('net', 'dropoutProb'))
  dropoutInputProb = ast.literal_eval(cfg.get('net', 'dropoutInputProb'))
  wLenLimit = ast.literal_eval(cfg.get('net', 'wLenLimit'))
  momentumInitial = ast.literal_eval(cfg.get('net', 'momentumInitial'))
  momentumFinal = ast.literal_eval(cfg.get('net', 'momentumFinal'))
  momentumT = ast.literal_eval(cfg.get('net', 'momentumT'))

  mlp = MLP.MLP(layerSizes, 
      activations, 
      doDropout=doDropout, 
      dropoutType=dropoutType, 
      dropoutProb=dropoutProb, 
      dropoutInputProb=dropoutInputProb,
      wLenLimit=wLenLimit)

  # Additionally load the experiment parameters
  digits = list(ast.literal_eval(cfg.get('experiment', 'digits')))
  mnistPath = cfg.get('experiment', 'mnistPath')
  numEpochs = ast.literal_eval(cfg.get('experiment', 'numEpochs'))
  minibatchSize = ast.literal_eval(cfg.get('experiment', 'minibatchSize'))
  learningRate = ast.literal_eval(cfg.get('experiment', 'learningRate'))
  rateDecay = ast.literal_eval(cfg.get('experiment', 'rateDecay'))
  checkGradient = ast.literal_eval(cfg.get('experiment', 'checkGradient'))

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
    shutil.copyfile(sys.argv[1], paramsCopyStr)

  # Load the corresponding data
  X_tr, Y_tr, X_te, Y_te = bdp.loadMNISTnp(mnistPath, digits=digits,
      asBitVector=True)

  print "Training for "+str(numEpochs)+" epochs:"
  p = momentumInitial
  for t in range(0, numEpochs):
    if t == 2:
      if checkGradient:
        print "Checking gradient..."
        mlp.check_gradient(X_tr[0:10, :], Y_tr[0:10, :], learningRate, 0)
        print " Gradient checking complete."

    startTime = time.time()

    for i in range(0, X_tr.shape[0], minibatchSize):
      mlp.train(X_tr[i:i+minibatchSize, :], Y_tr[i:i+minibatchSize],
          learningRate, p)


      if i%(1*minibatchSize)==0:
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
      f.flush()

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

