import numpy as np
import theano
import theano.tensor as T

import lasagne
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from theano.gpuarray import dnn
from theano.gpuarray.type import gpuarray_shared_constructor
from config import mode_with_gpu, mode_without_gpu, test_ctx_name, ref_cast
from theano.configdefaults import SUPPORTED_DNN_CONV_ALGO_FWD
from six import StringIO

from lasagne.layers import Layer
from lasagne.nonlinearities import identity, softmax, rectify, softplus, sigmoid, softmax, tanh
from lasagne import init
import numpy as np
import lasagne
import pickle as pkl
import logging
import os
import uuid
from os.path import join, exists, abspath
from os import makedirs, listdir
import time
import datetime
import logging
import sys
import os
import time
from collections import OrderedDict
import scipy.io as sio
import theano

def reversematrix(org_matrix):
	col, row = org_matrix.shape
	reverse_matrix = np.zeros((col, row)).astype('float32')
	for i in xrange(col):
		for j in xrange(row):
			reverse_matrix[col-1-j,row-1-i] = org_matrix[i,j]
	return reverse_matrix

def _log_dot_matrix(x, z):
    y = x[:, :, None] + z[None, :, :]
    y_max = y.max(axis=1)
    out = T.log(T.sum(T.exp(y - y_max[:, None, :]), axis=1)) + y_max
    return T.switch(T.isnan(out), -numpy.inf, out)

def _log_add(a, b):
	# TODO: move functions like this to utils
	max_ = T.maximum(a, b)
	result = T.cast(max_ + T.log(1+T.exp(a + b - 2 * max_)),dtype=theano.config.floatX)
	return T.switch(T.isnan(result), max_, result)

def _log_batched_dot(x, z):
	x = T.cast(x, dtype=theano.config.floatX)
	z = T.cast(z, dtype=theano.config.floatX)
	y = x + z
	max_y = T.max((y),axis=1)
	out = T.log(T.sum(T.exp(y - max_y[:,None,:]), axis=1)) + max_y
	out = T.cast(out, dtype=theano.config.floatX)
	return T.switch(T.isnan(out), -np.inf, out)

def f_logsumexp(A, axis=None):
	"""Numerically stable log( sum( exp(A) ) ) """
	A_max = T.max(A, axis=axis, keepdims=True)
	B = T.log(T.sum(T.exp(A-A_max), axis=axis, keepdims=True, dtype=theano.config.floatX))+A_max
	#B = T.sum(B, axis=axis, dtype=theano.config.floatX)
	return B

def f_sumexp(A, axis=None):
	"""Numerically stable log( sum( exp(A) ) ) """
	A_max = T.max(A, axis=axis, keepdims=True)
	B = T.log(T.sum(T.exp(A-A_max), axis=axis, keepdims=True, dtype=theano.config.floatX))+A_max
	#B = T.sum(B, axis=axis, dtype=theano.config.floatX)
	return B

def log_softmax(X, axis=None):
	k = T.max(X, axis=axis, keepdims=True)
	norm_X = X - k
	log_sum_exp_X = T.log(T.sum(T.exp(norm_X), axis=axis, keepdims=True, dtype=theano.config.floatX))
	output = T.cast(norm_X - log_sum_exp_X, dtype=theano.config.floatX)
	return output

def epslog(x):
	inf = T.cast(1E30, dtype=theano.config.floatX)
	min_inf = T.cast(1E-30, dtype=theano.config.floatX)
	return T.cast(T.log(T.clip(x, min_inf, inf)), theano.config.floatX)

def log_add(a, b):
	a = T.cast(a, dtype=theano.config.floatX)
	b = T.cast(b, dtype=theano.config.floatX)
	max_ = T.maximum(a, b)
	return T.cast(max_ + T.log(1+T.exp(a + b - 2 * max_)),dtype=theano.config.floatX)

def log_dot_matrix(x, z):
	x = T.cast(x, dtype=theano.config.floatX)
	z = T.cast(z, dtype=theano.config.floatX)
	inf = T.cast(1E30,dtype=theano.config.floatX)
	log_dot = T.dot(x, z)
	zeros_to_minus_inf = (z.max(axis=0) - 1) * inf
	return T.cast(log_dot + zeros_to_minus_inf,dtype=theano.config.floatX)


def log_dot_tensor(x, z):
	inf = T.cast(1E30,dtype=theano.config.floatX)
	x = T.cast(x, dtype=theano.config.floatX)
	z = T.cast(z, dtype=theano.config.floatX)
	log_dot = (x.dimshuffle(1, 'x', 0) * z).sum(axis=0).T
	zeros_to_minus_inf = (z.max(axis=0) - 1) * inf
	return T.cast(log_dot + zeros_to_minus_inf.T, dtype=theano.config.floatX)

def log_batched_dot(x, z):
	inf = T.cast(1E30,dtype=theano.config.floatX)
	x = T.cast(x, dtype=theano.config.floatX)
	z = T.cast(z, dtype=theano.config.floatX)
	batched_dot = T.sum(x*z,axis=1, dtype=theano.config.floatX, acc_dtype=theano.config.floatX)
	zeros_to_minus_inf = (z.max(axis=1) - 1) * inf
	return T.cast(batched_dot + zeros_to_minus_inf, dtype=theano.config.floatX)

def load_model(id, model_name, nonlinearitytype ,gamma, alpha, model_params_all, model_num):
	"""
	Load the pickled version of the model into a 'new' model instance.
	:param id: The model ID is constructed from the timestamp when the model was defined.
	"""
	model_name = model_name + ' ' + nonlinearitytype
	model_params = (model_name, gamma, alpha, id)
	load_model_params = (model_name, model_num, model_num, id)
	root = get_root_output_path(*model_params)
	p = get_model_path(root, *load_model_params[1:3])
	print 'model path ', p
	model_params = pkl.load(open(p, "rb"))
	for i in range(len(model_params_all)):
		init_param = model_params_all[i]
		loaded_param = model_params[i]
		print loaded_param.shape
		if not loaded_param.shape == tuple(init_param.shape.eval()):
			print "Model could not be loaded, since parameters are not aligned."
		model_params_all[i].set_value(np.asarray(model_params[i], dtype=theano.config.floatX), borrow=True)
	print '------------Succeeded in loading model-----------------'

def reinitialization_model(model_params, initializa_params):
	for param in model_params:
		if param.name in initializa_params.keys():
			param.set_value(np.asarray(initializa_params[param.name], 
							dtype=theano.config.floatX), borrow=True)
			print '--------set ', param.name, ' with new value'

	
	

def load_mat_model(id, model_name, nonlinearitytype ,gamma, alpha, model_params_all, epoch):
	model_name = model_name + ' ' + nonlinearitytype
	model_params = (model_name, gamma, alpha, id)
	root = get_root_output_path(*model_params)
	p = get_model_mat_path(root, epoch, epoch)
	p += 'model_params'
	model_params = sio.loadmat(p)
	for i in range(len(model_params_all)):
		init_param = model_params_all[i]
		loaded_param = model_params[init_param.name+'_'+str(i)]
		if loaded_param.shape[1] == 1 or loaded_param.shape[0] == 1:
			loaded_param = loaded_param.flatten()
		name = init_param.name+'_'+str(i)
		print name
		if not loaded_param.shape == tuple(init_param.shape.eval()):
			print "Model could not be loaded, since parameters are not aligned."
		model_params_all[i].set_value(np.asarray(loaded_param, dtype=theano.config.floatX), borrow=True)
	print '------------Succeeded in loading model-----------------'

def get_model_mat_path(root_path, gamma, alpha):
	return join(root_path, '%s_%s.mat' % (str(gamma), str(alpha)))

def pkl2mat(root_path, gamma, alpha, model_params_all):
	p = get_model_mat_path(root_path, gamma, alpha)
	p += 'model_params'
	if model_params_all is None:
		raise ("Model params are not set and can therefore not be .mat for matlab format.")
	params_dict = OrderedDict()
	k = 0
	for param in model_params_all:
		params_dict[param.name+'_'+str(k)] = param.get_value()
		k += 1
	sio.savemat(p,params_dict)
	print 'Succeeded in saving model parameters as .mat format for matlab'

def get_model_path(root_path, gamma, alpha):
	return join(root_path, '%s_%s.pkl' % (str(gamma), str(alpha)))

def dump_model(root_path, gamma, alpha, params):
	"""
	Dump the model into a pickled version in the model path formulated in the initialisation method.
	"""
	p = get_model_path(root_path, gamma, alpha)
	if params is None:
		raise ("Model params are not set and can therefore not be pickled.")
	params = [param.get_value() for param in params]
	pkl.dump(params, open(p, "wb"), protocol=pkl.HIGHEST_PROTOCOL)

def get_socres_path(root_path, name, gamma, alpha):
	return join(root_path, '%s_%s_%s.txt' % (str(name), str(gamma), str(alpha)))

def dump_cosine_dist(root_path, gamma, alpha, cosine_dist):
	"""
	Dump the model into a pickled version in the model path formulated in the initialisation method.
	"""
	score_path = get_socres_path(root_path, gamma, alpha)
	np.savetxt(score_path,cosine_dist)
	return score_path

def get_model_mat_path(root_path, gamma, alpha):
	return join(root_path, '%s_%s.mat' % (str(gamma), str(alpha)))

def path_exists(path):
	if not exists(path):
		makedirs(path)
	return path

def create_root_output_path(type, gamma, alpha):
	t = time.time()
	d = datetime.datetime.fromtimestamp(t).strftime('%Y%m%d%H%M%S')
	root = 'id_%s_%s_%s_%s' % (str(d), type, str(gamma), str(alpha))
	path = join(get_output_path(), root)
	if exists(path): path += "_(%s)" % str(uuid.uuid4())
	return path_exists(path)

def get_root_output_path(type, gamma, alpha, id):
	root = 'id_%s_%s_%s_%s' % (str(id), type, str(gamma), str(alpha))
	path = join(get_output_path(), root)
	return path

def get_logging_path(root_path):
	t = time.time()
	n = "_logging_%s.log" % datetime.datetime.fromtimestamp(t).strftime('%Y-%m-%d-%H%M%S')
	return join(root_path, n)

def get_output_path():
	#full_path = abspath('.')
	full_path = '/mnt/workspace/xuht/markvo-ctc'
	path = join(full_path, 'output')
	return path_exists(path)

def init_logging(model_name, nonlinearitytype, centor_loss_ratio, alpha):
	model_name = model_name + ' ' + nonlinearitytype
	logger = logging.getLogger('%slogger' % model_name)
	for hdlr in logger.handlers: logger.removeHandler(hdlr)
	rootpaths = create_root_output_path(model_name, centor_loss_ratio, alpha)
	hdlr = logging.FileHandler(get_logging_path(rootpaths))
	formatter = logging.Formatter('%(message)s')
	hdlr.setFormatter(formatter)
	logger.addHandler(hdlr)
	ch = logging.StreamHandler(sys.stdout)
	ch.setFormatter(formatter)
	logger.addHandler(ch)
	logger.setLevel(logging.INFO)
	return logger, rootpaths

def write_to_logger(logger, s):
	"""
	Write a string to the logger and the console.
	:param s: A string with the text to print.
	"""
	logger.info(s)


def txt_write(fobject, seg_name, lists):
	fobject.write(seg_name)
	fobject.write(' ')
	for i in xrange(len(lists)):
		fobject.write(str(lists[i]))
		fobject.write(' ')
	fobject.write('\n')

