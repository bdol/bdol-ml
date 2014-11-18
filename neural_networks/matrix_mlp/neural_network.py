"""
An implementation of a multilayer perceptron using matrix multiplies.

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

import numpy as np

def f_sigmoid(X, deriv=False):
    if not deriv:
        return 1 / (1 + np.exp(-X))
    else:
        return f_sigmoid(X)*(1 - f_sigmoid(X))


def f_softmax(X):
    Z = np.sum(np.exp(X), axis=1)
    Z = Z.reshape(Z.shape[0], 1)
    return np.exp(X) / Z


class Layer:
    def __init__(self, size, minibatch_size, is_input=False, is_output=False,
                 activation=f_sigmoid):
        self.is_input = is_input
        self.is_output = is_output

        # Z is the matrix that holds output values
        self.Z = np.zeros((minibatch_size, size[0]))
        # The activation function is an externally defined function (with a
        # derivative) that is stored here
        self.activation = activation

        # W is the outgoing weight matrix for this layer
        self.W = None
        # S is the matrix that holds the inputs to this layer
        self.S = None
        # D is the matrix that holds the deltas for this layer
        self.D = None
        # Fp is the matrix that holds the derivatives of the activation function
        self.Fp = None

        if not is_input:
            self.S = np.zeros((minibatch_size, size[0]))
            self.D = np.zeros((minibatch_size, size[0]))

        if not is_output:
            self.W = np.random.normal(size=size, scale=1E-4)

        if not is_input and not is_output:
            self.Fp = np.zeros((size[0], minibatch_size))

    def forward_propagate(self):
        if self.is_input:
            return self.Z.dot(self.W)

        self.Z = self.activation(self.S)
        if self.is_output:
            return self.Z
        else:
            # For hidden layers, we add the bias values here
            self.Z = np.append(self.Z, np.ones((self.Z.shape[0], 1)), axis=1)
            self.Fp = self.activation(self.S, deriv=True).T
            return self.Z.dot(self.W)


class MLP:
    def __init__(self, layer_config, minibatch_size=100):
        self.layers = []
        self.num_layers = len(layer_config)
        self.minibatch_size = minibatch_size

        for i in range(self.num_layers-1):
            if i == 0:
                print "Initializing input layer with size {0}.".format(
                    layer_config[i]
                )
                # Here, we add an additional unit at the input for the bias
                # weight.
                self.layers.append(Layer([layer_config[i]+1, layer_config[i+1]],
                                         minibatch_size,
                                         is_input=True))
            else:
                print "Initializing hidden layer with size {0}.".format(
                    layer_config[i]
                )
                # Here we add an additional unit in the hidden layers for the
                # bias weight.
                self.layers.append(Layer([layer_config[i]+1, layer_config[i+1]],
                                         minibatch_size,
                                         activation=f_sigmoid))

        print "Initializing output layer with size {0}.".format(
            layer_config[-1]
        )
        self.layers.append(Layer([layer_config[-1], None],
                                 minibatch_size,
                                 is_output=True,
                                 activation=f_softmax))
        print "Done!"

    def forward_propagate(self, data):
        # We need to be sure to add bias values to the input
        self.layers[0].Z = np.append(data, np.ones((data.shape[0], 1)), axis=1)

        for i in range(self.num_layers-1):
            self.layers[i+1].S = self.layers[i].forward_propagate()
        return self.layers[-1].forward_propagate()

    def backpropagate(self, yhat, labels):
        self.layers[-1].D = (yhat - labels).T
        for i in range(self.num_layers-2, 0, -1):
            # We do not calculate deltas for the bias values
            W_nobias = self.layers[i].W[0:-1, :]

            self.layers[i].D = W_nobias.dot(self.layers[i+1].D) * \
                               self.layers[i].Fp

    def update_weights(self, eta):
        for i in range(0, self.num_layers-1):
            W_grad = -eta*(self.layers[i+1].D.dot(self.layers[i].Z)).T
            self.layers[i].W += W_grad

    def evaluate(self, train_data, train_labels, test_data, test_labels,
                 num_epochs=500, eta=0.05, eval_train=False, eval_test=True):

        N_train = len(train_labels)*len(train_labels[0])
        N_test = len(test_labels)*len(test_labels[0])

        print "Training for {0} epochs...".format(num_epochs)
        for t in range(0, num_epochs):
            out_str = "[{0:4d}] ".format(t)

            for b_data, b_labels in zip(train_data, train_labels):
                output = self.forward_propagate(b_data)
                self.backpropagate(output, b_labels)
                self.update_weights(eta=eta)

            if eval_train:
                errs = 0
                for b_data, b_labels in zip(train_data, train_labels):
                    output = self.forward_propagate(b_data)
                    yhat = np.argmax(output, axis=1)
                    errs += np.sum(1-b_labels[np.arange(len(b_labels)), yhat])

                out_str = "{0} Training error: {1:.5f}".format(out_str,
                                                           float(errs)/N_train)

            if eval_test:
                errs = 0
                for b_data, b_labels in zip(test_data, test_labels):
                    output = self.forward_propagate(b_data)
                    yhat = np.argmax(output, axis=1)
                    errs += np.sum(1-b_labels[np.arange(len(b_labels)), yhat])

                out_str = "{0} Test error: {1:.5f}".format(out_str,
                                                       float(errs)/N_test)

            print out_str
