import ConfigParser
import cPickle as pickle
import argparse
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
  parser = argparse.ArgumentParser()
  parser.add_argument("-i", "--ini_file", type=str, help="initialize with the given .ini file")
  parser.add_argument("-c", "--continue_experiment", action="store_true", help="continue with the given model and experiment files")
  #parser.add_argument("-m", "--model_file", type=str, help="the model pickle to be used")
  parser.add_argument("-p", "--param_file", type=str, help="the parameter pickle to be used")
  parser.add_argument("-e", "--total_epochs", type=int, help="the new total epoch limit to use")

  args = parser.parse_args()
  model = None
  params = None

  if args.ini_file:
    params = MLP.Parameters(args.ini_file)
    print "Initialized new experiment and model with parameters:"
    print "=============================================================="
    print params

    model = MLP.MLP(params)
    params.model = model


  elif args.continue_experiment:
    #if not args.model_file:
      #print "Error: continue experiment specified, but no model file given!"
      #sys.exit(1)
    if not args.param_file:
      print "Error: continue experiment specified, but no parameter file given!"
      sys.exit(1)
      
    print "Loading {0}...".format(args.param_file)
    params = pickle.load(open(args.param_file, "rb"))
    params.initLog()
    model = params.model

    print "Loaded existing experiment and model with parameters:"
    print "=============================================================="
    print params

    if args.total_epochs:
      params.totalEpochs = total_epochs
      print "Updated total number of experiment epochs to {0}.".format(total_epochs)

  else:
    parser.print_help()
    sys.exit(1)


  # Additionally load the experiment parameters
  # Load MNIST data
  digits = params.digits
  mnistPath = params.datasetPath 
  X_tr, Y_tr, X_te, Y_te = bdp.loadMNISTnp(mnistPath, digits=digits,
      asBitVector=True)

  for t in range(params.currentEpoch, params.totalEpochs):
    startTime = time.time()
    for i in range(0, X_tr.shape[0], params.minibatchSize):
      model.train(X_tr[i:i+params.minibatchSize, :], Y_tr[i:i+params.minibatchSize])
      bdp.progBar(i, X_tr.shape[0])
    bdp.progBar(X_tr.shape[0], X_tr.shape[0])
    
    elapsedTime = time.time() - startTime
    print " Epoch {0}, learning rate: {1:.4f}, momentum: {2:.4f} elapsed time:{3:.2f}s".format(params.currentEpoch, 
        params.currentLearningRate, params.momentumCurrent, elapsedTime)

    # Compute the training and testing errors
    YhatTrain = model.test(X_tr)
    numErrsTrain = numErrs(Y_tr, YhatTrain)
    YhatTest = model.test(X_te)
    numErrsTest = numErrs(Y_te, YhatTest)

    errsStr = "Train errors: {0}\n".format(numErrsTrain)
    errsStr += "Test errors: {0}\n".format(numErrsTest)
    print errsStr

    logStr = "{0},{1},{2},{3},{4},{5:.2f}\n".format(
                params.currentEpoch,numErrsTrain, numErrsTest, params.currentLearningRate,
                params.momentumCurrent, elapsedTime)
    if params.logToFile:
      params.log(logStr)

    params.update()

  params.cleanup()
