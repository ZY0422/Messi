import numpy as np
import theano
import theano.tensor as T

import lasagne
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
#from theano.sandbox.cuda.rng_curand import CURAND_RandomStreams as RandomStreams

from theano.gpuarray import dnn
from theano.gpuarray.type import gpuarray_shared_constructor
from config import mode_with_gpu, mode_without_gpu, test_ctx_name, ref_cast
from theano.configdefaults import SUPPORTED_DNN_CONV_ALGO_FWD
from six import StringIO

import gzip

from lasagne.layers import Layer
from lasagne.nonlinearities import identity, softmax, rectify, softplus, sigmoid, softmax, tanh
from lasagne import init
from utils import (f_logsumexp, log_softmax, epslog, 
					log_add, log_dot_matrix, 
					log_dot_tensor) 
# from markov_ctc_layer import ListIndexLayer, TopologyCTCLayer

np.random.seed(10)

def BRNNConcatLayer(input_layer, n_hidden, grad_clip, nonlinearity):
	l_fwd = lasagne.layers.RecurrentLayer(input_layer, n_hidden,
					W_in_to_hid=init.Uniform(0.1),
					W_hid_to_hid=init.Uniform(0.1),
					b=init.Constant(0.),
					nonlinearity=nonlinearity,
					hid_init=init.Constant(0.),
					backwards=False,
					grad_clipping=grad_clip,
					learn_init=False)

	l_bwd = lasagne.layers.RecurrentLayer(input_layer, n_hidden,
					W_in_to_hid=init.Uniform(0.1),
					W_hid_to_hid=init.Uniform(0.1),
					b=init.Constant(0.),
					nonlinearity=nonlinearity,
					hid_init=init.Constant(0.),
					backwards=True,
					grad_clipping=grad_clip,
					learn_init=False)

	return lasagne.layers.ConcatLayer([l_fwd,l_bwd],axis=2)

def BGRUConcatLayer(input_layer, n_hidden, grad_clip):
	resetgate=lasagne.layers.recurrent.Gate(
		W_in=lasagne.init.Normal(0.1), W_hid=lasagne.init.Normal(0.1), \
		W_cell=None, b=lasagne.init.Constant(0.0))
	updategate=lasagne.layers.recurrent.Gate(
		W_in=lasagne.init.Normal(0.1), W_hid=lasagne.init.Normal(0.1), \
		W_cell=None, b=lasagne.init.Constant(0.0))        
	hidden_update=lasagne.layers.recurrent.Gate(
		W_in=lasagne.init.Normal(0.1), W_hid=lasagne.init.Normal(0.1), \
		W_cell=None, b=lasagne.init.Constant(0.0), nonlinearity=lasagne.nonlinearities.tanh)

	l_fwd = lasagne.layers.GRULayer(
		input_layer, n_hidden, backwards=False, resetgate=resetgate, 
		updategate=updategate, 
		hidden_update=hidden_update,
		grad_clipping=grad_clip,
		learn_init=False)

	resetgate=lasagne.layers.recurrent.Gate(
		W_in=lasagne.init.Normal(0.1), W_hid=lasagne.init.Normal(0.1), \
		W_cell=None, b=lasagne.init.Constant(0.0))
	updategate=lasagne.layers.recurrent.Gate(
		W_in=lasagne.init.Normal(0.1), W_hid=lasagne.init.Normal(0.1), \
		W_cell=None, b=lasagne.init.Constant(0.0))        
	hidden_update=lasagne.layers.recurrent.Gate(
		W_in=lasagne.init.Normal(0.1), W_hid=lasagne.init.Normal(0.1), \
		W_cell=None, b=lasagne.init.Constant(0.0), nonlinearity=lasagne.nonlinearities.tanh)
    
	l_bwd = lasagne.layers.GRULayer(
		input_layer, n_hidden, backwards=True, resetgate=resetgate, 
		updategate=updategate, 
		hidden_update=hidden_update,
		grad_clipping=grad_clip,
		learn_init=False)

	return lasagne.layers.ConcatLayer((l_fwd,l_bwd),axis=2)

def BLSTMConcatLayer_normal(input_layer, n_hidden, grad_clip, nonlinearity):
	l_fwd = lasagne.layers.LSTMLayer(
					input_layer, n_hidden, backwards=False, 
					grad_clipping=grad_clip,
					nonlinearity=nonlinearity,
					peepholes=False)

	l_bwd = lasagne.layers.LSTMLayer(
					input_layer, n_hidden, backwards=True,  
					grad_clipping=grad_clip,
					nonlinearity=nonlinearity,
					peepholes=False)
	
	return lasagne.layers.ConcatLayer([l_fwd,l_bwd],axis=2)

def BLSTMConcatLayer(input_layer, n_hidden, grad_clip, nonlinearity):
	"""
	This function generates a BLSTM by concatenating a forward and a backward LSTM
	at axis 2, which should be the axis for the hidden dimension (batch_size x seq_len x hidden_dim)
	:parameters: See LSTMLayer for inputs, this layer receives the same inputs as a LSTM-Layer
	:returns: lasagne.layers.ConcatLayer of 2 LSTM layers 
	"""

	ingate=lasagne.layers.recurrent.Gate(
		W_in=lasagne.init.Uniform(0.1), W_hid=lasagne.init.Uniform(0.1), \
		W_cell=lasagne.init.Uniform(0.1), b=lasagne.init.Uniform(0.1))
	forgetgate=lasagne.layers.recurrent.Gate(
		W_in=lasagne.init.Uniform(0.1), W_hid=lasagne.init.Uniform(0.1), \
		W_cell=lasagne.init.Uniform(0.1), b=lasagne.init.Uniform(0.1))        
	outgate=lasagne.layers.recurrent.Gate(
		W_in=lasagne.init.Uniform(0.1), W_hid=lasagne.init.Uniform(0.1), \
		W_cell=lasagne.init.Uniform(0.1), b=lasagne.init.Uniform(0.1))
	cell=lasagne.layers.recurrent.Gate(
		W_in=lasagne.init.Uniform(0.1), W_hid=lasagne.init.Uniform(0.1), \
		W_cell=None, b=lasagne.init.Uniform(0.1), nonlinearity=lasagne.nonlinearities.tanh)
    
	l_fwd = lasagne.layers.LSTMLayer(
		input_layer, n_hidden, backwards=False, ingate=ingate, 
		forgetgate=forgetgate, 
		cell=cell, outgate=outgate, 
		grad_clipping=grad_clip,
		nonlinearity=nonlinearity,
		learn_init=False,
		peepholes=True,
		cell_init=lasagne.init.Constant(0.0),
		hid_init=lasagne.init.Constant(0.0))

	ingate=lasagne.layers.recurrent.Gate(
		W_in=lasagne.init.Uniform(0.1), W_hid=lasagne.init.Uniform(0.1), \
		W_cell=lasagne.init.Uniform(0.1), b=lasagne.init.Uniform(0.1))
	forgetgate=lasagne.layers.recurrent.Gate(
		W_in=lasagne.init.Uniform(0.1), W_hid=lasagne.init.Uniform(0.1), \
		W_cell=lasagne.init.Uniform(0.1), b=lasagne.init.Uniform(0.1))        
	outgate=lasagne.layers.recurrent.Gate(
		W_in=lasagne.init.Uniform(0.1), W_hid=lasagne.init.Uniform(0.1), \
		W_cell=lasagne.init.Uniform(0.1), b=lasagne.init.Uniform(0.1))
	cell=lasagne.layers.recurrent.Gate(
		W_in=lasagne.init.Uniform(0.1), W_hid=lasagne.init.Uniform(0.1), \
		W_cell=None, b=lasagne.init.Uniform(0.1), nonlinearity=lasagne.nonlinearities.tanh)
	
	l_bwd = lasagne.layers.LSTMLayer(
		input_layer, n_hidden, backwards=True,  ingate=ingate, 
		forgetgate=forgetgate, cell=cell, grad_clipping=grad_clip,
		outgate=outgate,
		nonlinearity=nonlinearity,
		learn_init=False,
		peepholes=True,
		cell_init=lasagne.init.Constant(0.0),
		hid_init=lasagne.init.Constant(0.0))
	
	return lasagne.layers.ConcatLayer((l_fwd,l_bwd),axis=2)

def UniLSTMLayer(input_layer, n_hidden, grad_clip, nonlinearity):
	ingate=lasagne.layers.recurrent.Gate(
		W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(), \
		W_cell=lasagne.init.Uniform(0.1), b=lasagne.init.Constant(0.0))
	forgetgate=lasagne.layers.recurrent.Gate(
		W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(), \
		W_cell=lasagne.init.Uniform(0.1), b=lasagne.init.Constant(1.0))        
	outgate=lasagne.layers.recurrent.Gate(
		W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(), \
		W_cell=lasagne.init.Uniform(0.1), b=lasagne.init.Constant(0.0))
	cell=lasagne.layers.recurrent.Gate(
		W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(), \
		W_cell=None, b=lasagne.init.Constant(0.0), nonlinearity=lasagne.nonlinearities.tanh)
    
	l = lasagne.layers.LSTMLayer(
		input_layer, n_hidden, backwards=False, ingate=ingate, 
		forgetgate=forgetgate, 
		cell=cell, outgate=outgate, 
		grad_clipping=grad_clip,
		nonlinearity=nonlinearity,
		learn_init=False,
		peepholes=True,
		cell_init=lasagne.init.Constant(0.0),
		hid_init=lasagne.init.Constant(0.0))
	return l

def BLSTMConcatLayer_1(input_layer, n_hidden, grad_clip, nonlinearity):
	"""
	This function generates a BLSTM by concatenating a forward and a backward LSTM
	at axis 2, which should be the axis for the hidden dimension (batch_size x seq_len x hidden_dim)
	:parameters: See LSTMLayer for inputs, this layer receives the same inputs as a LSTM-Layer
	:returns: lasagne.layers.ConcatLayer of 2 LSTM layers 
	"""

	ingate=lasagne.layers.recurrent.Gate(
		W_in=lasagne.init.Normal(0.1), W_hid=lasagne.init.Normal(0.1), \
		W_cell=lasagne.init.Normal(0.1), b=lasagne.init.Constant(0.0))
	forgetgate=lasagne.layers.recurrent.Gate(
		W_in=lasagne.init.Normal(0.1), W_hid=lasagne.init.Normal(0.1), \
		W_cell=lasagne.init.Normal(0.1), b=lasagne.init.Constant(0.0))        
	outgate=lasagne.layers.recurrent.Gate(
		W_in=lasagne.init.Normal(0.1), W_hid=lasagne.init.Normal(0.1), \
		W_cell=lasagne.init.Normal(0.1), b=lasagne.init.Constant(0.0))
	cell=lasagne.layers.recurrent.Gate(
		W_in=lasagne.init.Normal(0.1), W_hid=lasagne.init.Normal(0.1), \
		W_cell=None, b=lasagne.init.Constant(0.0), nonlinearity=lasagne.nonlinearities.tanh)
    
	l_fwd = lasagne.layers.LSTMLayer(
		input_layer, n_hidden, backwards=False, ingate=ingate, 
		forgetgate=forgetgate, 
		cell=cell, outgate=outgate, 
		grad_clipping=grad_clip,
		nonlinearity=nonlinearity,
		learn_init=False,
		peepholes=False,
		cell_init=lasagne.init.Constant(0.0),
		hid_init=lasagne.init.Constant(0.0))

	ingate=lasagne.layers.recurrent.Gate(
		W_in=lasagne.init.Normal(0.1), W_hid=lasagne.init.Normal(0.1), \
		W_cell=lasagne.init.Normal(0.1), b=lasagne.init.Constant(0.0))
	forgetgate=lasagne.layers.recurrent.Gate(
		W_in=lasagne.init.Normal(0.1), W_hid=lasagne.init.Normal(0.1), \
		W_cell=lasagne.init.Normal(0.1), b=lasagne.init.Constant(0.0))        
	outgate=lasagne.layers.recurrent.Gate(
		W_in=lasagne.init.Normal(0.1), W_hid=lasagne.init.Normal(0.1), \
		W_cell=lasagne.init.Normal(0.1), b=lasagne.init.Constant(0.0))
	cell=lasagne.layers.recurrent.Gate(
		W_in=lasagne.init.Normal(0.1), W_hid=lasagne.init.Normal(0.1), \
		W_cell=None, b=lasagne.init.Constant(0.0), nonlinearity=lasagne.nonlinearities.tanh)
    
	l_bwd = lasagne.layers.LSTMLayer(
		input_layer, n_hidden, backwards=True,  ingate=ingate, 
		forgetgate=forgetgate, cell=cell, grad_clipping=grad_clip,
		outgate=outgate,
		nonlinearity=nonlinearity,
		learn_init=False,
		peepholes=False,
		cell_init=lasagne.init.Constant(0.0),
		hid_init=lasagne.init.Constant(0.0))
	
	return lasagne.layers.ConcatLayer((l_fwd,l_bwd),axis=2)

class TransitionLayer(lasagne.layers.Layer):
	# generate all transition kernles parallely
	def __init__(self, incoming, num_units, num_labels,
		Wh=init.Normal(0.1), Wy=init.Normal(0.1), 
		bs=init.Constant(0.0), Wy_init=True, nonlinearity=lasagne.nonlinearities.sigmoid,
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
		self.Wy_init = Wy_init

		self.Wh = self.add_param(Wh, (self.num_inputs, self.num_units), name='h_to_s')
		self.Wy = self.add_param(Wy, (self.num_labels, self.num_units), name='y_to_s', trainable=self.Wy_init)
		self.bs = self.add_param(bs, (self.num_units,), name="bs")

	def get_output_shape_for(self, input_shape):
		return (self.n_batch, self.n_seq, self.num_labels, self.num_units)
    
	def get_output_for(self, input, **kwargs):

		batch, seq_lens, data_dim = input.shape
		
		zeros = T.zeros_like(self.Wy)
		inputs = input.dimshuffle((0,1,'x',2)) # shuffle to batch * seq * num_labels * data_dim
		
		output = self.nonlinearity(T.dot(inputs[:,1:,:], self.Wh) + self.Wy.dimshuffle('x','x',0,1) + self.bs)
		output_first_frame = self.nonlinearity(T.dot(inputs[:,0,:].dimshuffle(0,'x',1,2), self.Wh) + zeros.dimshuffle('x','x',0,1) + self.bs)
		outputs = T.concatenate([output_first_frame,output],axis=(1))
		# the 2-dim is previous label and 3-dim is current label just like a_ij of hmm that
		# transition probability from i to j
		
		return  outputs

class SoftmaxTransitionLayer(lasagne.layers.Layer):
	# generate all transition kernles parallely
	def __init__(self, incoming, num_units, num_labels,
		Wh=init.Normal(0.1), Wy=init.Normal(0.1), 
		bs=init.Constant(0.0), Wy_init=True, nonlinearity=lasagne.nonlinearities.softmax,
		**kwargs):
		super(SoftmaxTransitionLayer, self).__init__(incoming, **kwargs)

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
		self.Wy_init = Wy_init

		self.Wh = self.add_param(Wh, (self.num_inputs, self.num_units), name='h_to_s')
		self.Wy = self.add_param(Wy, (self.num_labels, self.num_units), name='y_to_s', trainable=self.Wy_init)
		self.bs = self.add_param(bs, (self.num_units,), name="bs")

	def get_output_shape_for(self, input_shape):
		return (self.n_batch, self.n_seq, self.num_labels, self.num_units)
    
	def get_output_for(self, input, **kwargs):

		batch, seq_lens, data_dim = input.shape
		
		zeros = T.zeros_like(self.Wy)
		inputs = input.dimshuffle((0,1,'x',2)) # shuffle to batch * seq * num_labels * data_dim
		
		output = (T.dot(inputs[:,1:,:], self.Wh) + self.Wy.dimshuffle('x','x',0,1) + self.bs)
		output_first_frame = (T.dot(inputs[:,0,:].dimshuffle(0,'x',1,2), self.Wh) + zeros.dimshuffle('x','x',0,1) + self.bs)
		outputs = T.concatenate([output_first_frame,output],axis=(1))
		log_classifier = log_softmax(outputs, axis=-1)

		# the 2-dim is previous label and 3-dim is current label just like a_ij of hmm that
		# transition probability from i to j
		
		return  log_classifier

class RecurrentTransitionLayer(lasagne.layers.Layer):
	# generate all transition kernles parallely
	def __init__(self, incoming, num_units, num_labels,
		Wh=init.Normal(0.1), Wy=init.Normal(0.1), 
		Ws=init.Normal(0.1), hid_init=init.Constant(0.0), b=init.Constant(0.0), 
		Wy_init=True, nonlinearity=lasagne.nonlinearities.sigmoid,
		**kwargs):
		super(RecurrentTransitionLayer, self).__init__(incoming, **kwargs)

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
		self.Wy_init = Wy_init

		self.hid_init = self.add_param(
				hid_init, (1, self.num_units), name="hid_init",
				trainable=False, regularizable=False)

		self.Wh = self.add_param(Wh, (self.num_units, self.num_units), name='h_to_s')
		self.Ws = self.add_param(Ws, (self.num_inputs, self.num_units), name='s_to_s')
		self.Wy = self.add_param(Wy, (self.num_labels, self.num_units), name='y_to_s', trainable=self.Wy_init)
		self.b = self.add_param(b, (self.num_units,), name="bs")

	def get_output_shape_for(self, input_shape):
		return (self.n_batch, self.n_seq, self.num_labels, self.num_units)
    
	def get_output_for(self, input, **kwargs):

		batch, seq_lens, data_dim = input.shape

		ones = T.ones((batch, 1))
		hid_init = T.dot(ones, self.hid_init) # batch x data_dim
		hid_init = hid_init.dimshuffle((0,1,'x'))
		hid_init = T.dot(hid_init, T.ones((1, self.num_labels))) # batch x dim x labels
		hid_init = hid_init.dimshuffle((0,2,1)) # shuffle to batch x labels x dim

		zeros = T.zeros_like(self.Wy)
		inputs = input.dimshuffle((1,0,'x',2)) # shuffle to batch * seq * num_labels * data_dim

		# pre-compute
		pre_compute = (T.dot(inputs[1:,:], self.Ws) + self.Wy.dimshuffle('x','x',0,1) + self.b)
		pre_compute_first_frame = (T.dot(inputs[0,:].dimshuffle('x',0,1,2), self.Ws) + zeros.dimshuffle('x','x',0,1) + self.b)
		pre_compute_output = T.concatenate([pre_compute_first_frame,pre_compute],axis=(0))

		def step(input_n, hid_previous):
			output = T.dot(hid_previous, self.Wh) + input_n
			return self.nonlinearity(output)

		hid_out, _ = theano.scan(fn=step,
					sequences=[pre_compute_output],
					outputs_info=[hid_init])

		hid_out = hid_out.dimshuffle((1,0,2,3))
		
		return  hid_out

class RecurrentSoftmaxTransitionLayer(lasagne.layers.Layer):
	# generate all transition kernles parallely
	def __init__(self, incoming, num_units, num_labels,
		Wh=init.Normal(0.1), Wy=init.Normal(0.1), 
		Ws=init.Normal(0.1), hid_init=init.Constant(0.0), b=init.Constant(0.0), 
		Wy_init=True, nonlinearity=lasagne.nonlinearities.softmax,
		**kwargs):
		super(RecurrentSoftmaxTransitionLayer, self).__init__(incoming, **kwargs)

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
		self.Wy_init = Wy_init

		self.hid_init = self.add_param(
				hid_init, (1, self.num_units), name="hid_init",
				trainable=False, regularizable=False)

		self.Wh = self.add_param(Wh, (self.num_units, self.num_units), name='h_to_s')
		self.Ws = self.add_param(Ws, (self.num_inputs, self.num_units), name='s_to_s')
		self.Wy = self.add_param(Wy, (self.num_labels, self.num_units), name='y_to_s', trainable=self.Wy_init)
		self.b = self.add_param(b, (self.num_units,), name="bs")

	def get_output_shape_for(self, input_shape):
		return (self.n_batch, self.n_seq, self.num_labels, self.num_units)
    
	def get_output_for(self, input, **kwargs):

		batch, seq_lens, data_dim = input.shape

		ones = T.ones((batch, 1))
		hid_init = T.dot(ones, self.hid_init) # batch x data_dim
		hid_init = hid_init.dimshuffle((0,1,'x'))
		hid_init = T.dot(hid_init, T.ones((1, self.num_labels))) # batch x dim x labels
		hid_init = hid_init.dimshuffle((0,2,1)) # shuffle to batch x labels x dim

		zeros = T.zeros_like(self.Wy)
		inputs = input.dimshuffle((1,0,'x',2)) # shuffle to batch * seq * num_labels * data_dim
		
		# pre-compute
		pre_compute = (T.dot(inputs[1:,:], self.Ws) + self.Wy.dimshuffle('x','x',0,1) + self.b)
		pre_compute_first_frame = (T.dot(inputs[0,:].dimshuffle('x',0,1,2), self.Ws) + zeros.dimshuffle('x','x',0,1) + self.b)
		pre_compute_output = T.concatenate([pre_compute_first_frame,pre_compute],axis=(0))

		def step(input_n, hid_previous):
			output = T.dot(hid_previous, self.Wh) + input_n
			return log_softmax(output)

		hid_out, _ = theano.scan(fn=step,
					sequences=[pre_compute_output],
					outputs_info=[hid_init])
		hid_out = hid_out.dimshuffle((1,0,2,3))
		
		return  hid_out

class BatchDenseLayer(lasagne.layers.Layer):
	# generate all transition kernles parallely
	def __init__(self, incoming, num_units,
		W=init.Normal(0.1), b=init.Constant(0.0), nonlinearity=lasagne.nonlinearities.softplus,
		**kwargs):
		super(BatchDenseLayer, self).__init__(incoming, **kwargs)

		self.nonlinearity = (lasagne.nonlinearities.identity if nonlinearity is None
							else nonlinearity)

		self.num_inputs = self.input_shape[-1] # data dimension
		self.n_seq = self.input_shape[1] # seq_length
		self.n_batch = self.input_shape[0] # batch size

		self.num_units = num_units

		self.W = self.add_param(W, (self.num_inputs, self.num_units), name='w')
		self.b = self.add_param(b, (self.num_units,), name="b")

	def get_output_shape_for(self, input_shape):
		return (self.n_batch, self.n_seq, self.num_units) 

	def get_output_for(self, input, **kwargs):
		batch, seq, data_dim = input.shape
		linear = T.dot(input, self.W) + self.b
		output = self.nonlinearity(linear)
		return output

class BatchSoftmaxLayer(lasagne.layers.Layer):
	# generate all transition kernles parallely
	def __init__(self, incoming, num_labels,
		W=init.Normal(0.1), b=init.Constant(0.0), nonlinearity=lasagne.nonlinearities.softmax,
		**kwargs):
		super(BatchSoftmaxLayer, self).__init__(incoming, **kwargs) 
		self.num_inputs = self.input_shape[-1] # data dimension
		self.n_seq = self.input_shape[1] # seq_length
		self.n_batch = self.input_shape[0] # batch size

		self.num_labels = num_labels

		self.W = self.add_param(W, (self.num_inputs, self.num_labels), name='w')
		self.b = self.add_param(b, (self.num_labels,), name="b")

	def get_output_shape_for(self, input_shape):
		return (self.n_batch, self.n_seq, self.num_labels, self.num_labels) 

	def get_output_for(self, input, **kwargs):
		batch, seq, label_dim, data_dim = input.shape
		linear = T.dot(input, self.W) + self.b
		log_classifier = log_softmax(linear, axis=-1)
		return (log_classifier)

class BatchSoftmaxLinearLayer(lasagne.layers.Layer):
	# generate all transition kernles parallely
	def __init__(self, incoming, num_labels,
		W=init.Normal(0.1), b=init.Constant(0.0), nonlinearity=lasagne.nonlinearities.softmax,
		**kwargs):
		super(BatchSoftmaxLinearLayer, self).__init__(incoming, **kwargs) 
		self.num_inputs = self.input_shape[-1] # data dimension
		self.n_seq = self.input_shape[1] # seq_length
		self.n_batch = self.input_shape[0] # batch size

		self.num_labels = num_labels

		self.W = self.add_param(W, (self.num_inputs, self.num_labels), name='w')
		self.b = self.add_param(b, (self.num_labels,), name="b")

	def get_output_shape_for(self, input_shape):
		return (self.n_batch, self.n_seq, self.num_labels, self.num_labels) 

	def get_output_for(self, input, **kwargs):
		batch, seq, label_dim, data_dim = input.shape
		linear = T.dot(input, self.W) + self.b
		
		return (linear)

def test_model(input_var, input_dim, n_hidden, grad_clip,
				num_units, num_labels, nonlinearity_type,Wy_init):
	input_layer = lasagne.layers.InputLayer(shape=(None, None, input_dim),input_var=input_var)
	network = RecurrentTransitionLayer(input_layer, num_units, num_labels, 
									Wh=lasagne.init.GlorotUniform(), 
									Wy=lasagne.init.GlorotUniform(), 
									Ws=lasagne.init.GlorotUniform(),
									hid_init=init.Constant(0.0),
									b=init.Constant(0.0), 
									Wy_init=Wy_init,
									nonlinearity=nonlinearity_type)
	return network


def build_linear_model(input_var, input_dim, n_hidden, grad_clip,
				num_units, num_labels, nonlinearity_type,Wy_init):
	# input dimension is batch x seq x data-dim
	input_layer = lasagne.layers.InputLayer(shape=(None, None, input_dim),input_var=input_var)

	# network = UniLSTMLayer(input_layer, n_hidden, grad_clip, lasagne.nonlinearities.tanh)
	# network = UniLSTMLayer(network, n_hidden, grad_clip, lasagne.nonlinearities.tanh)
	# network = UniLSTMLayer(network, n_hidden, grad_clip, lasagne.nonlinearities.tanh)

	# network = BLSTMConcatLayer(input_layer, n_hidden, grad_clip, lasagne.nonlinearities.tanh)
	# network = BLSTMConcatLayer(network, n_hidden, grad_clip, lasagne.nonlinearities.tanh)
	# network = BLSTMConcatLayer(network, n_hidden, grad_clip, lasagne.nonlinearities.tanh)

	network = BLSTMConcatLayer_1(input_layer, n_hidden, grad_clip, lasagne.nonlinearities.tanh)
	network = BLSTMConcatLayer_1(network, n_hidden, grad_clip, lasagne.nonlinearities.tanh)
	network = BLSTMConcatLayer_1(network, n_hidden, grad_clip, lasagne.nonlinearities.tanh)

	# network = BGRUConcatLayer(input_layer, n_hidden, grad_clip)
	# network = BGRUConcatLayer(network, n_hidden, grad_clip)
	# network = BGRUConcatLayer(network, n_hidden, grad_clip)

	# network = BRNNConcatLayer(input_layer, n_hidden, grad_clip, lasagne.nonlinearities.tanh)
	# network = BRNNConcatLayer(network, n_hidden, grad_clip, lasagne.nonlinearities.tanh)
	# network = BRNNConcatLayer(network, n_hidden, grad_clip, lasagne.nonlinearities.tanh)

	# network = BLSTMConcatLayer_normal(input_layer, n_hidden, grad_clip, lasagne.nonlinearities.tanh)
	# network = BLSTMConcatLayer_normal(network, n_hidden, grad_clip, lasagne.nonlinearities.tanh)
	# network = BLSTMConcatLayer_normal(network, n_hidden, grad_clip, lasagne.nonlinearities.tanh)

	# batch * seq_len * data_dim
	transition_kernel = SoftmaxTransitionLayer(network, num_labels, num_labels, 
							Wh=lasagne.init.GlorotUniform(), 
							Wy=lasagne.init.GlorotUniform(), 
							bs=init.Constant(0.0), 
							Wy_init=Wy_init,
							nonlinearity=nonlinearity_type)

	# output of transition_softmax is batch * seq * num_labels * num_labels

	return transition_kernel

def build_dropout_model(input_var, input_dim, n_hidden, grad_clip,
				num_units, num_labels, nonlinearity_type,Wy_init,
				drop_ratio):
	# input dimension is batch x seq x data-dim
	input_layer = lasagne.layers.InputLayer(shape=(None, None, input_dim),input_var=input_var)

	#network = BLSTMConcatLayer_1(input_layer, n_hidden, grad_clip, lasagne.nonlinearities.tanh)

	network = BLSTMConcatLayer(input_layer, n_hidden, grad_clip, lasagne.nonlinearities.tanh)
	#network = BGRUConcatLayer(input_layer, n_hidden, grad_clip)
	network_dropout = lasagne.layers.DropoutLayer(network, p=drop_ratio, rescale=True)

	#network = BLSTMConcatLayer_1(network_dropout, n_hidden, grad_clip, lasagne.nonlinearities.tanh)
	network = BLSTMConcatLayer(network_dropout, n_hidden, grad_clip, lasagne.nonlinearities.tanh)
	#network = BGRUConcatLayer(network_dropout, n_hidden, grad_clip)
	network_dropout = lasagne.layers.DropoutLayer(network, p=drop_ratio, rescale=True)

	#network = BLSTMConcatLayer_1(network_dropout, n_hidden, grad_clip, lasagne.nonlinearities.tanh)
	network = BLSTMConcatLayer(network_dropout, n_hidden, grad_clip, lasagne.nonlinearities.tanh)
	#network = BGRUConcatLayer(network_dropout, n_hidden, grad_clip)
	network_dropout = lasagne.layers.DropoutLayer(network, p=drop_ratio, rescale=True)
	
	# batch * seq_len * data_dim
	network = TransitionLayer(network_dropout, num_units, num_labels, 
							Wh=lasagne.init.Uniform(0.1), 
							Wy=lasagne.init.Uniform(0.1), 
							bs=lasagne.init.Uniform(0.1), 
							Wy_init=Wy_init,
							nonlinearity=nonlinearity_type)

	# output of network is batch * seq * num_labels * num_units
	transition_kernel = BatchSoftmaxLayer(network, num_labels, 
						W=lasagne.init.Uniform(0.1), b=lasagne.init.Uniform(0.1))

	# output of transition_softmax is batch * seq * num_labels * num_labels

	return transition_kernel

def build_test_dropout_error_signal_model(input_var, input_dim, n_hidden, grad_clip,
				num_units, num_labels, nonlinearity_type,Wy_init):
	# input dimension is batch x seq x data-dim
	input_layer = lasagne.layers.InputLayer(shape=(None, None, input_dim),input_var=input_var)

	#network = BLSTMConcatLayer_1(input_layer, n_hidden, grad_clip, lasagne.nonlinearities.tanh)

	network = BLSTMConcatLayer(input_layer, n_hidden, grad_clip, lasagne.nonlinearities.tanh)
	#network = BGRUConcatLayer(input_layer, n_hidden, grad_clip)
	network_dropout = lasagne.layers.DropoutLayer(network, p=0.5)

	#network = BLSTMConcatLayer_1(network_dropout, n_hidden, grad_clip, lasagne.nonlinearities.tanh)
	network = BLSTMConcatLayer(network_dropout, n_hidden, grad_clip, lasagne.nonlinearities.tanh)
	#network = BGRUConcatLayer(network_dropout, n_hidden, grad_clip)
	network_dropout = lasagne.layers.DropoutLayer(network, p=0.5)

	#network = BLSTMConcatLayer_1(network_dropout, n_hidden, grad_clip, lasagne.nonlinearities.tanh)
	network = BLSTMConcatLayer(network_dropout, n_hidden, grad_clip, lasagne.nonlinearities.tanh)
	#network = BGRUConcatLayer(network_dropout, n_hidden, grad_clip)
	network_dropout = lasagne.layers.DropoutLayer(network, p=0.5)
	
	# batch * seq_len * data_dim
	network = TransitionLayer(network_dropout, num_units, num_labels, 
							Wh=lasagne.init.Uniform(0.1), 
							Wy=lasagne.init.Uniform(0.1), 
							bs=lasagne.init.Constant(0.0), 
							Wy_init=Wy_init,
							nonlinearity=nonlinearity_type)

	# output of network is batch * seq * num_labels * num_units
	transition_kernel = BatchSoftmaxLinearLayer(network, num_labels, 
						W=lasagne.init.Uniform(0.1), b=lasagne.init.Constant(0.0))

	# output of transition_softmax is batch * seq * num_labels * num_labels

	return transition_kernel

def build_5dropout_model(input_var, input_dim, n_hidden, grad_clip,
				num_units, num_labels, nonlinearity_type,Wy_init, drop_ratio):
	# input dimension is batch x seq x data-dim
	input_layer = lasagne.layers.InputLayer(shape=(None, None, input_dim),input_var=input_var)

	network = BLSTMConcatLayer(input_layer, n_hidden, grad_clip, lasagne.nonlinearities.tanh)
	network_dropout = lasagne.layers.DropoutLayer(network, p=drop_ratio)

	network = BLSTMConcatLayer(network_dropout, n_hidden, grad_clip, lasagne.nonlinearities.tanh)
	network_dropout = lasagne.layers.DropoutLayer(network, p=drop_ratio)

	network = BLSTMConcatLayer(network_dropout, n_hidden, grad_clip, lasagne.nonlinearities.tanh)
	network_dropout = lasagne.layers.DropoutLayer(network, p=drop_ratio)

	network = BLSTMConcatLayer(network_dropout, n_hidden, grad_clip, lasagne.nonlinearities.tanh)
	network_dropout = lasagne.layers.DropoutLayer(network, p=drop_ratio)

	network = BLSTMConcatLayer(network_dropout, n_hidden, grad_clip, lasagne.nonlinearities.tanh)
	network_dropout = lasagne.layers.DropoutLayer(network, p=drop_ratio)
	
	# batch * seq_len * data_dim
	network = TransitionLayer(network_dropout, num_units, num_labels, 
							Wh=lasagne.init.Uniform(0.1), 
							Wy=lasagne.init.Uniform(0.1), 
							bs=lasagne.init.Uniform(0.1), 
							Wy_init=Wy_init,
							nonlinearity=nonlinearity_type)

	# output of network is batch * seq * num_labels * num_units
	transition_kernel = BatchSoftmaxLayer(network, num_labels, 
						W=lasagne.init.Uniform(0.1), b=lasagne.init.Uniform(0.1))

	# output of transition_softmax is batch * seq * num_labels * num_labels

	return transition_kernel

def build_model(input_var, input_dim, n_hidden, grad_clip,
				num_units, num_labels, nonlinearity_type,Wy_init):
	# input dimension is batch x seq x data-dim
	input_layer = lasagne.layers.InputLayer(shape=(None, None, input_dim),input_var=input_var)

	network = BLSTMConcatLayer(input_layer, n_hidden, grad_clip, lasagne.nonlinearities.tanh)
	network = BLSTMConcatLayer(network, n_hidden, grad_clip, lasagne.nonlinearities.tanh)
	network = BLSTMConcatLayer(network, n_hidden, grad_clip, lasagne.nonlinearities.tanh)

	# batch * seq_len * data_dim
	network = TransitionLayer(network, num_units, num_labels, 
							Wh=lasagne.init.Uniform(0.1), 
							Wy=lasagne.init.Uniform(0.1), 
							bs=lasagne.init.Uniform(0.1), 
							Wy_init=Wy_init,
							nonlinearity=nonlinearity_type)

	# output of network is batch * seq * num_labels * num_units
	transition_kernel = BatchSoftmaxLayer(network, num_labels, 
						W=lasagne.init.Uniform(0.1), b=lasagne.init.Uniform(0.1))

	# output of transition_softmax is batch * seq * num_labels * num_labels

	return transition_kernel

def build_recurrent_model(input_var, input_dim, n_hidden, grad_clip,
				num_units, num_labels, nonlinearity_type,Wy_init):
	# input dimension is batch x seq x data-dim
	input_layer = lasagne.layers.InputLayer(shape=(None, None, input_dim),input_var=input_var)

	# network = UniLSTMLayer(input_layer, n_hidden, grad_clip, lasagne.nonlinearities.tanh)
	# network = UniLSTMLayer(network, n_hidden, grad_clip, lasagne.nonlinearities.tanh)
	# network = UniLSTMLayer(network, n_hidden, grad_clip, lasagne.nonlinearities.tanh)

	network = BLSTMConcatLayer(input_layer, n_hidden, grad_clip, lasagne.nonlinearities.tanh)
	network = BLSTMConcatLayer(network, n_hidden, grad_clip, lasagne.nonlinearities.tanh)
	network = BLSTMConcatLayer(network, n_hidden, grad_clip, lasagne.nonlinearities.tanh)

	# network = BLSTMConcatLayer_1(input_layer, n_hidden, grad_clip, lasagne.nonlinearities.tanh)
	# network = BLSTMConcatLayer_1(network, n_hidden, grad_clip, lasagne.nonlinearities.tanh)
	# network = BLSTMConcatLayer_1(network, n_hidden, grad_clip, lasagne.nonlinearities.tanh)

	# network = BGRUConcatLayer(input_layer, n_hidden, grad_clip)
	# network = BGRUConcatLayer(network, n_hidden, grad_clip)
	# network = BGRUConcatLayer(network, n_hidden, grad_clip)

	# network = BRNNConcatLayer(input_layer, n_hidden, grad_clip, lasagne.nonlinearities.tanh)
	# network = BRNNConcatLayer(network, n_hidden, grad_clip, lasagne.nonlinearities.tanh)
	# network = BRNNConcatLayer(network, n_hidden, grad_clip, lasagne.nonlinearities.tanh)

	# network = BLSTMConcatLayer_normal(input_layer, n_hidden, grad_clip, lasagne.nonlinearities.tanh)
	# network = BLSTMConcatLayer_normal(network, n_hidden, grad_clip, lasagne.nonlinearities.tanh)
	# network = BLSTMConcatLayer_normal(network, n_hidden, grad_clip, lasagne.nonlinearities.tanh)

	# batch * seq_len * data_dim
	network = RecurrentTransitionLayer(network, num_units, num_labels, 
									Wh=lasagne.init.GlorotUniform(), 
									Wy=lasagne.init.GlorotUniform(), 
									Ws=lasagne.init.GlorotUniform(),
									hid_init=init.Constant(0.0),
									b=init.Constant(0.0), 
									Wy_init=Wy_init,
									nonlinearity=nonlinearity_type)

	# output of network is batch * seq * num_labels * num_units
	transition_kernel = BatchSoftmaxLayer(network, num_labels, 
						W=lasagne.init.GlorotUniform(), b=init.Constant(0.0))

	# output of transition_softmax is batch * seq * num_labels * num_labels

	return transition_kernel

def build_recurrent_dropout_model(input_var, input_dim, n_hidden, grad_clip,
				num_units, num_labels, nonlinearity_type,Wy_init):
	# input dimension is batch x seq x data-dim
	input_layer = lasagne.layers.InputLayer(shape=(None, None, input_dim),input_var=input_var)

	# network = UniLSTMLayer(input_layer, n_hidden, grad_clip, lasagne.nonlinearities.tanh)
	# network = UniLSTMLayer(network, n_hidden, grad_clip, lasagne.nonlinearities.tanh)
	# network = UniLSTMLayer(network, n_hidden, grad_clip, lasagne.nonlinearities.tanh)

	network = BLSTMConcatLayer(input_layer, n_hidden, grad_clip, lasagne.nonlinearities.tanh)
	network_dropout = lasagne.layers.DropoutLayer(network, p=0.3)
	network = BLSTMConcatLayer(network_dropout, n_hidden, grad_clip, lasagne.nonlinearities.tanh)
	network_dropout = lasagne.layers.DropoutLayer(network, p=0.3)
	network = BLSTMConcatLayer(network_dropout, n_hidden, grad_clip, lasagne.nonlinearities.tanh)
	network_dropout = lasagne.layers.DropoutLayer(network, p=0.3)
	# network = BLSTMConcatLayer_1(input_layer, n_hidden, grad_clip, lasagne.nonlinearities.tanh)
	# network = BLSTMConcatLayer_1(network, n_hidden, grad_clip, lasagne.nonlinearities.tanh)
	# network = BLSTMConcatLayer_1(network, n_hidden, grad_clip, lasagne.nonlinearities.tanh)

	# network = BGRUConcatLayer(input_layer, n_hidden, grad_clip)
	# network = BGRUConcatLayer(network, n_hidden, grad_clip)
	# network = BGRUConcatLayer(network, n_hidden, grad_clip)

	# network = BRNNConcatLayer(input_layer, n_hidden, grad_clip, lasagne.nonlinearities.tanh)
	# network = BRNNConcatLayer(network, n_hidden, grad_clip, lasagne.nonlinearities.tanh)
	# network = BRNNConcatLayer(network, n_hidden, grad_clip, lasagne.nonlinearities.tanh)

	# network = BLSTMConcatLayer_normal(input_layer, n_hidden, grad_clip, lasagne.nonlinearities.tanh)
	# network = BLSTMConcatLayer_normal(network, n_hidden, grad_clip, lasagne.nonlinearities.tanh)
	# network = BLSTMConcatLayer_normal(network, n_hidden, grad_clip, lasagne.nonlinearities.tanh)

	# batch * seq_len * data_dim
	network = RecurrentTransitionLayer(network_dropout, num_units, num_labels, 
									Wh=lasagne.init.Uniform(0.1), 
									Wy=lasagne.init.Uniform(0.1), 
									Ws=lasagne.init.Uniform(0.1),
									hid_init=init.Constant(0.0),
									b=init.Constant(0.0), 
									Wy_init=Wy_init,
									nonlinearity=nonlinearity_type)

	# output of network is batch * seq * num_labels * num_units
	transition_kernel = BatchSoftmaxLayer(network, num_labels, 
						W=lasagne.init.Uniform(0.1), b=init.Constant(0.0))

	# output of transition_softmax is batch * seq * num_labels * num_labels

	return transition_kernel

def build_recurrent_linear_model(input_var, input_dim, n_hidden, grad_clip,
				num_units, num_labels, nonlinearity_type,Wy_init):
	# input dimension is batch x seq x data-dim
	input_layer = lasagne.layers.InputLayer(shape=(None, None, input_dim),input_var=input_var)

	# network = UniLSTMLayer(input_layer, n_hidden, grad_clip, lasagne.nonlinearities.tanh)
	# network = UniLSTMLayer(network, n_hidden, grad_clip, lasagne.nonlinearities.tanh)
	# network = UniLSTMLayer(network, n_hidden, grad_clip, lasagne.nonlinearities.tanh)

	network = BLSTMConcatLayer(input_layer, n_hidden, grad_clip, lasagne.nonlinearities.tanh)
	network = BLSTMConcatLayer(network, n_hidden, grad_clip, lasagne.nonlinearities.tanh)
	network = BLSTMConcatLayer(network, n_hidden, grad_clip, lasagne.nonlinearities.tanh)

	# network = BLSTMConcatLayer_1(input_layer, n_hidden, grad_clip, lasagne.nonlinearities.tanh)
	# network = BLSTMConcatLayer_1(network, n_hidden, grad_clip, lasagne.nonlinearities.tanh)
	# network = BLSTMConcatLayer_1(network, n_hidden, grad_clip, lasagne.nonlinearities.tanh)

	# network = BGRUConcatLayer(input_layer, n_hidden, grad_clip)
	# network = BGRUConcatLayer(network, n_hidden, grad_clip)
	# network = BGRUConcatLayer(network, n_hidden, grad_clip)

	# network = BRNNConcatLayer(input_layer, n_hidden, grad_clip, lasagne.nonlinearities.tanh)
	# network = BRNNConcatLayer(network, n_hidden, grad_clip, lasagne.nonlinearities.tanh)
	# network = BRNNConcatLayer(network, n_hidden, grad_clip, lasagne.nonlinearities.tanh)

	# network = BLSTMConcatLayer_normal(input_layer, n_hidden, grad_clip, lasagne.nonlinearities.tanh)
	# network = BLSTMConcatLayer_normal(network, n_hidden, grad_clip, lasagne.nonlinearities.tanh)
	# network = BLSTMConcatLayer_normal(network, n_hidden, grad_clip, lasagne.nonlinearities.tanh)

	# batch * seq_len * data_dim
	network = RecurrentSoftmaxTransitionLayer(network, num_labels, num_labels, 
									Wh=lasagne.init.GlorotUniform(), 
									Wy=lasagne.init.GlorotUniform(), 
									Ws=lasagne.init.GlorotUniform(),
									hid_init=init.Constant(0.0),
									b=init.Constant(0.0), 
									Wy_init=Wy_init,
									nonlinearity=nonlinearity_type)

	# output of transition_softmax is batch * seq * num_labels * num_labels

	return transition_kernel

def build_toy_model(input_var, input_dim, n_hidden, grad_clip,
				num_units, num_labels, nonlinearity):
	# input dimension is batch x seq x data-dim
	input_layer = lasagne.layers.InputLayer(shape=(None, None, input_dim),input_var=input_var)

	network = BLSTMConcatLayer(input_layer, n_hidden, grad_clip, nonlinearity)
	network = BLSTMConcatLayer(network, n_hidden, grad_clip, nonlinearity)
	network = BLSTMConcatLayer(network, n_hidden, grad_clip, nonlinearity)

	# batch * seq_len * data_dim
	tran = TransitionLayer(network, num_units, num_labels, Wh=lasagne.init.GlorotUniform(), 
							Wy=lasagne.init.GlorotUniform(), bs=init.Constant(0.0), Wy_init=True,
							nonlinearity=lasagne.nonlinearities.tanh)

	# output of network is batch * seq * num_labels * num_units
	transition_kernel = BatchSoftmaxLayer(tran, num_labels, 
						W=lasagne.init.GlorotUniform(), b=init.Constant(0.0))

	return transition_kernel, network, tran

def build_toy_recurrent_model(input_var, input_dim, n_hidden, grad_clip,
				num_units, num_labels, nonlinearity):
	# input dimension is batch x seq x data-dim
	input_layer = lasagne.layers.InputLayer(shape=(None, None, input_dim),input_var=input_var)

	network = BLSTMConcatLayer(input_layer, n_hidden, grad_clip, nonlinearity)
	network = BLSTMConcatLayer(network, n_hidden, grad_clip, nonlinearity)
	network = BLSTMConcatLayer(network, n_hidden, grad_clip, nonlinearity)

	# batch * seq_len * data_dim
	tran = RecurrentTransitionLayer(network, num_units, num_labels, 
									Wh=lasagne.init.GlorotUniform(), 
									Wy=lasagne.init.GlorotUniform(), 
									Ws=lasagne.init.GlorotUniform(),
									hid_init=init.Constant(0.0),
									b=init.Constant(0.0), 
									Wy_init=True,
									nonlinearity=nonlinearity)

	# output of network is batch * seq * num_labels * num_units
	transition_kernel = BatchSoftmaxLayer(tran, num_labels, 
						W=lasagne.init.GlorotUniform(), b=init.Constant(0.0))

	return transition_kernel, network, tran

def build_toy_linear_model(input_var, input_dim, n_hidden, grad_clip,
				num_units, num_labels, nonlinearity):
	# input dimension is batch x seq x data-dim
	input_layer = lasagne.layers.InputLayer(shape=(None, None, input_dim),input_var=input_var)

	network = BLSTMConcatLayer(input_layer, n_hidden, grad_clip, nonlinearity)
	network = BLSTMConcatLayer(network, n_hidden, grad_clip, nonlinearity)
	network = BLSTMConcatLayer(network, n_hidden, grad_clip, nonlinearity)

	# output of network is batch * seq * num_labels * num_units
	transition_kernel = SoftmaxTransitionLayer(network, num_units, num_labels, 
							Wh=lasagne.init.GlorotUniform(), 
							Wy=lasagne.init.GlorotUniform(), 
							bs=init.Constant(0.0), 
							Wy_init=True,
							nonlinearity=lasagne.nonlinearities.tanh)

	return transition_kernel, network, transition_kernel

def build_toy_ctc_model(input_var, input_dim, n_hidden, grad_clip,
                num_units, num_labels, nonlinearity):
    # input dimension is batch x seq x data-dim
    input_layer = lasagne.layers.InputLayer(shape=(None, None, input_dim),input_var=input_var)

    network = BLSTMConcatLayer(input_layer, n_hidden, grad_clip, nonlinearity)
    network = BLSTMConcatLayer(network, n_hidden, grad_clip, nonlinearity)
    network = BLSTMConcatLayer(network, n_hidden, grad_clip, nonlinearity)

    transition_kernel = lasagne.layers.RecurrentLayer(network, num_labels, 
                                nonlinearity=lasagne.nonlinearities.softmax,
                                grad_clipping=grad_clip)

    return transition_kernel

def build_ctc_model(input_var, input_dim, n_hidden, grad_clip,
				num_units, num_labels, nonlinearity_type):
	# input dimension is batch x seq x data-dim
	input_layer = lasagne.layers.InputLayer(shape=(None, None, input_dim),input_var=input_var)

	network = BLSTMConcatLayer_1(input_layer, n_hidden, grad_clip, lasagne.nonlinearities.tanh)
	network = BLSTMConcatLayer_1(network, n_hidden, grad_clip, lasagne.nonlinearities.tanh)
	network = BLSTMConcatLayer_1(network, n_hidden, grad_clip, lasagne.nonlinearities.tanh)

	network = BatchDenseLayer(network, num_units=num_units,
								W=lasagne.init.GlorotUniform(),
								b=lasagne.init.Constant(0.0),
								nonlinearity=nonlinearity_type)

	transition_kernel = BatchDenseLayer(network, num_units=num_labels,
								W=lasagne.init.GlorotUniform(),
								b=lasagne.init.Constant(0.0),
								nonlinearity=lasagne.nonlinearities.linear)

	return transition_kernel

# def build_topologyctclayer_model(input_var, input_label, input_mask,
# 								input_dim, n_hidden, grad_clip,
# 								num_units, num_labels, nonlinearity_type,Wy_init):
	
# 	input_data_layer = lasagne.layers.InputLayer(shape=(None, None, input_dim),input_var=input_var)


def build_warp_ctc_dropout_model(input_var, input_dim, n_hidden,
	grad_clip, num_units, num_labels, nonlinearity_type):
	input_layer = lasagne.layers.InputLayer(shape=(None, None, input_dim),input_var=input_var)

	network = BLSTMConcatLayer(input_layer, n_hidden, grad_clip, lasagne.nonlinearities.tanh)
	network_dropout = lasagne.layers.DropoutLayer(network, p=0.3)

	network = BLSTMConcatLayer(network_dropout, n_hidden, grad_clip, lasagne.nonlinearities.tanh)
	network_dropout = lasagne.layers.DropoutLayer(network, p=0.3)

	network = BLSTMConcatLayer(network_dropout, n_hidden, grad_clip, lasagne.nonlinearities.tanh)
	network_dropout = lasagne.layers.DropoutLayer(network, p=0.3)
	
	# batch * seq_len * data_dim
	network = BatchDenseLayer(network_dropout, num_units, 
							W=lasagne.init.Uniform(0.1), 
							b=lasagne.init.Uniform(0.1),
							nonlinearity=nonlinearity_type)

	transition_kernel = BatchDenseLayer(network, num_labels, 
						W=lasagne.init.Uniform(0.1), 
						b=lasagne.init.Uniform(0.1),
						nonlinearity=lasagne.nonlinearities.linear)

	# output is batch * seq * num_labels

	return transition_kernel


def bulid_ZYCnn(input_var=None):
	#input layer
	#Very important!!!
	#I need to know the size of the input data
	#None means batchsize,1 means channel, 28*28 means size
	l_in=lasagne.layers.InputLayer(shape=(None,1,28,28),
		input_var=input_var)
	
	l_conv1=lasagne.layers.Conv2DLayer(
		l_in,num_filters=128,filter_size=(3,5),
		nonlinearity=lasagne.nonlinearities.rectify,
		W=lasagne.init.GlorotUniform())
	#Only the first layer need maxpooling
	l_maxpool=lasagne.layers.MaxPool2DLayer(l_conv1,pool_size=(3,1))
	
	l_conv2=lasagne.layers.Conv2DLayer(
		l_maxpool,num_filters=128,filter_size=(3,5),
		nonlinearity=lasagne.nonlinearities.rectify)
	l_pool2=lasagne.layers.FeaturePoolLayer(
		l_conv2,pool_size=128,axis=1,pool_function=theano.tensor.max)
	l_conv3=lasagne.layers.Conv2DLayer(
		l_pool2,num_filters=128,filter_size=(3,5),
		nonlinearity=lasagne.nonlinearities.rectify)
	l_pool3=lasagne.layers.FeaturePoolLayer(
		l_conv3,pool_size=128,axis=1,pool_function=theano.tensor.max)
	l_conv4=lasagne.layers.Conv2DLayer(
		l_pool3,num_filters=128,filter_size=(3,5),
		nonlinearity=lasagne.nonlinearities.rectify)
	l_pool4=lasagne.layers.FeaturePoolLayer(
		l_conv4,pool_size=128,axis=1,pool_function=theano.tensor.max)
	l_conv5=lasagne.layers.Conv2DLayer(
		l_pool4,num_filters=32,filter_size=(3,5),
		nonlinearity=lasagne.nonlinearities.rectify)
	l_pool5=lasagne.layers.FeaturePoolLayer(
		l_conv5,pool_size=32,axis=1,pool_function=theano.tensor.max)
	l_conv6=lasagne.layers.Conv2DLayer(
		l_pool5,num_filters=32,filter_size=(3,5),
		nonlinearity=lasagne.nonlinearities.rectify)
	l_pool6=lasagne.layers.FeaturePoolLayer(
		l_conv6,pool_size=32,axis=1,pool_function=theano.tensor.max)
	l_conv7=lasagne.layers.Conv2DLayer(
		l_pool6,num_filters=32,filter_size=(3,5),
		nonlinearity=lasagne.nonlinearities.rectify)
	l_pool7=lasagne.layers.FeaturePoolLayer(
		l_conv7,pool_size=32,axis=1,pool_function=theano.tensor.max)
	l_conv8=lasagne.layers.Conv2DLayer(
		l_pool7,num_filters=32,filter_size=(3,5),
		nonlinearity=lasagne.nonlinearities.rectify)
	l_pool8=lasagne.layers.FeaturePoolLayer(
		l_conv8,pool_size=32,axis=1,pool_function=theano.tensor.max)
	l_conv9=lasagne.layers.Conv2DLayer(
		l_pool8,num_filters=32,filter_size=(3,5),
		nonlinearity=lasagne.nonlinearities.rectify)
	l_pool9=lasagne.layers.FeaturePoolLayer(
		l_conv9,pool_size=32,axis=1,pool_function=theano.tensor.max)
	l_conv10=lasagne.layers.Conv2DLayer(
		l_pool9,num_filters=32,filter_size=(3,5),
		nonlinearity=lasagne.nonlinearities.rectify)
	l_pool10=lasagne.layers.FeaturePoolLayer(
		l_conv10,pool_size=32,axis=1,pool_function=theano.tensor.max)

	l_dense1=lasagne.layers.DenseLayer(
		l_pool10,num_units=1024,
		nonlinearity=lasagne.nonlinearities.rectify,
		W=lasagne.init.GlorotUniform())
	l_densepool1=lasagne.layers.FeaturePoolLayer(
		l_dense1,pool_size=32,axis=1,pool_function=theano.tensor.max)
	l_dense2=lasagne.layers.DenseLayer(
		l_densepool1,num_units=1024,
		nonlinearity=lasagne.nonlinearities.rectify,
		W=lasagne.init.GlorotUniform())
	l_densepool2=lasagne.layers.FeaturePoolLayer(
		l_dense2,pool_size=32,axis=1,pool_function=theano.tensor.max)
	l_dense3=lasagne.layers.DenseLayer(
		l_densepool2,num_units=1024,
		nonlinearity=lasagne.nonlinearities.rectify,
		W=lasagne.init.GlorotUniform())
	l_densepool2=lasagne.layers.FeaturePoolLayer(
		l_dense3,pool_size=32,axis=1,pool_function=theano.tensor.max)

	l_ZYout=lasagne.layers.DenseLayer(
		lasagne.layers.dropout(l_densepool3,p=0.5),
		num_units=10,
		nonlinearity=lasagne.nonlinearities.softmax)	
	return l_ZYout