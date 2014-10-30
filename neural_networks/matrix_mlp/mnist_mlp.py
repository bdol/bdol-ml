import cPickle, gzip
from data_utils import create_minibatches
from neural_network import MLP

f = gzip.open('/home/bdol/data/mnist.pkl.gz')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

minibatch_size = 100
print "Creating data..."
train_data, train_labels = create_minibatches(train_set[0], train_set[1],
                                              minibatch_size,
                                              create_bit_vector=True)
valid_data, valid_labels = create_minibatches(valid_set[0], valid_set[1],
                                              minibatch_size,
                                              create_bit_vector=True)
print "Done!"


mlp = MLP(layer_config=[784, 100, 100, 10], minibatch_size=minibatch_size)
mlp.evaluate(train_data, train_labels, valid_data, valid_labels,
             eval_train=True)