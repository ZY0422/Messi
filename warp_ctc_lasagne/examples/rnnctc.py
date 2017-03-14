from __future__ import print_function

import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne import init
import gpu_ctc

class TransitionLayer(lasagne.layers.Layer):
    # generate all transition kernles parallely
    def __init__(self, incoming, num_units, num_labels,
        Wh=init.GlorotUniform(), bs=init.Constant(0.0), 
        nonlinearity=lasagne.nonlinearities.sigmoid,
        **kwargs):
        super(TransitionLayer, self).__init__(incoming, **kwargs)

        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                            else nonlinearity)
        # input dimension is batch * seq * num_labels * data_dim
        # first dimension shuffle output of lstm/rnn from batch * seq * data_dim to
        # batch * seq * num_labels * data_dim
        self.num_inputs = self.input_shape[-1] # data dimension
        self.n_seq = self.input_shape[1] # seq_length
        self.n_batch = self.input_shape[0] # batch size
        self.num_units = num_units
        self.num_labels = num_labels

        self.Wh = self.add_param(Wh, (self.num_inputs, self.num_units), name='h_to_s')
        self.bs = self.add_param(bs, (self.num_units,), name="bs")

    def get_output_shape_for(self, input_shape):
        return (self.n_batch, self.n_seq, self.num_units)
    
    def get_output_for(self, input, **kwargs):

        batch, seq_lens, data_dim = input.shape
        
        output = self.nonlinearity(T.dot(input, self.Wh) + self.bs)
        # the 2-dim is previous label and 3-dim is current label just like a_ij of hmm that
        # transition probability from i to j
        
        return  output

num_classes = 6
mbsz = 1
min_len = 12
max_len = 12
n_hidden = 100
grad_clip = 100

input_lens = T.ivector('input_lens')
output = T.ivector('output')
output_lens = T.ivector('output_lens')

l_in = lasagne.layers.InputLayer(shape=(mbsz, max_len, num_classes))

h1f = lasagne.layers.RecurrentLayer(l_in, n_hidden, grad_clipping=grad_clip,
        nonlinearity=lasagne.nonlinearities.tanh)
h1b = lasagne.layers.RecurrentLayer(l_in, n_hidden, grad_clipping=grad_clip,
        nonlinearity=lasagne.nonlinearities.tanh, backwards = True)
h1 = lasagne.layers.ConcatLayer([h1f, h1b],axis=2)

h2f = lasagne.layers.RecurrentLayer(h1, n_hidden, grad_clipping=grad_clip,
        nonlinearity=lasagne.nonlinearities.tanh)
h2b = lasagne.layers.RecurrentLayer(h1, n_hidden, grad_clipping=grad_clip,
        nonlinearity=lasagne.nonlinearities.tanh, backwards = True)
h2 = lasagne.layers.ConcatLayer([h2f, h2b],axis=2)

h3 = lasagne.layers.RecurrentLayer(h2, num_classes, grad_clipping=grad_clip,
        nonlinearity=lasagne.nonlinearities.linear)
# Turn <minibatch_size, max_length, num_classes> into <max_length, minibatch_size, num_classes>
l_out = lasagne.layers.DimshuffleLayer(h3, (1, 0, 2))

network_output = lasagne.layers.get_output(l_out)

cost = T.mean(gpu_ctc.ctc_gpu_costs(network_output, input_lens, output, output_lens))
grads = T.grad(cost, wrt=network_output)
all_params = lasagne.layers.get_all_params(l_out)
updates = lasagne.updates.adam(cost, all_params, 0.001)

train = theano.function([l_in.input_var, input_lens, output, output_lens], cost, updates=updates,
                        allow_input_downcast=True)
predict = theano.function([l_in.input_var], network_output,
                        allow_input_downcast=True)
get_grad = theano.function([l_in.input_var, input_lens, output, output_lens], grads,
                        allow_input_downcast=True)

from loader_simple import DataLoader
data_loader = DataLoader(mbsz=mbsz, min_len=min_len, max_len=max_len, num_classes=num_classes)

i = 1
while i<=10000:
    i += 1
    print(i)
    sample = data_loader.sample()
    cost = train(*sample)
    out = predict(sample[0])
    print(cost)
    print(sample[0].shape)
    print(sample[1])
    print(sample[2])
    print(sample[3])
    print("input", sample[0][0].argmax(1))
    print("prediction", out[:, 0].argmax(1))
    print("expected", sample[2][:sample[3][0]])


