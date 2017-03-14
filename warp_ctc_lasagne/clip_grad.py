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
from lasagne import utils
from lasagne.updates import get_or_compute_grads, norm_constraint, total_norm_constraint

theano_rng = RandomStreams()

def rmsprop_with_grad_clipping(loss_or_grads, params, learning_rate=1.0, 
								rho=0.9, epsilon=1e-6, rescale=1.0):
    
	grads = get_or_compute_grads(loss_or_grads, params)
	updates = OrderedDict()

	learning_rate = T.cast(learning_rate,dtype=theano.config.floatX)
	rho = T.cast(rho,dtype=theano.config.floatX)
	epsilon = T.cast(epsilon,dtype=theano.config.floatX)

	grad_norm = T.sqrt(sum(map(lambda x: T.sqr(x).sum(), grads)))
	not_finite = T.or_(T.isnan(grad_norm), T.isinf(grad_norm))
	grad_norm = T.sqrt(grad_norm)
	scaling_num = T.cast(rescale,dtype=theano.config.floatX)
	scaling_den = T.maximum(rescale, grad_norm)

    # Using theano constant to prevent upcasting of float32
	one = T.constant(1)

	for param, grad in zip(params, grads):
		value = param.get_value(borrow=True)
		accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
							broadcastable=param.broadcastable)

		grad = T.switch(not_finite, 0.1 * param, grad * (scaling_num / scaling_den))

		accu_new = rho * accu + (one - rho) * grad ** 2
		updates[accu] = accu_new
		updates[param] = param - (learning_rate * grad /
								T.sqrt(accu_new + epsilon))

	return updates

def adam_with_grad_clipping(loss_or_grads, params, learning_rate=0.001, beta1=0.9,
		beta2=0.999, epsilon=1e-8, rescale=1.0):
   
	all_grads = get_or_compute_grads(loss_or_grads, params)

	epsilon = T.cast(epsilon,dtype=theano.config.floatX)
	beta1 = T.cast(beta1,dtype=theano.config.floatX)
	beta2 = T.cast(beta2,dtype=theano.config.floatX)

	grad_norm = T.sqrt(sum(map(lambda x: T.sqr(x).sum(), all_grads)))
	not_finite = T.or_(T.isnan(grad_norm), T.isinf(grad_norm))
	grad_norm = T.sqrt(grad_norm)
	scaling_num = T.cast(rescale,dtype=theano.config.floatX)
	learning_rate = T.cast(learning_rate,dtype=theano.config.floatX)
	scaling_den = T.maximum(rescale, grad_norm)

	t_prev = theano.shared(utils.floatX(0.))
	updates = OrderedDict()

	t = t_prev + 1
	a_t = learning_rate*T.sqrt(1-beta2**t)/(1-beta1**t)

	for param, g_t in zip(params, all_grads):

		g_t = T.switch(not_finite, 0.1 * param, g_t * (scaling_num / scaling_den) )

		value = param.get_value(borrow=True)
		m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=param.broadcastable)
		v_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=param.broadcastable)

		m_t = beta1*m_prev + (1-beta1)*g_t
		v_t = beta2*v_prev + (1-beta2)*g_t**2
		step = a_t*m_t/(T.sqrt(v_t) + epsilon)

		updates[m_prev] = m_t
		updates[v_prev] = v_t
		updates[param] = param - step

	updates[t_prev] = t
	return updates

def adadelta_with_grad_clipping(loss_or_grads, params, learning_rate=1.0, rho=0.95, epsilon=1e-6, rescale=5.0):

	grads = get_or_compute_grads(loss_or_grads, params)
	updates = OrderedDict()

	grad_norm = T.sqrt(sum(map(lambda x: T.sqr(x).sum(), grads)))
	not_finite = T.or_(T.isnan(grad_norm), T.isinf(grad_norm))
	grad_norm = T.sqrt(grad_norm)
	scaling_num = rescale
	scaling_den = T.maximum(rescale, grad_norm)
    
	for param, grad in zip(params, grads):
		grad = T.switch(not_finite, 0.1 * param, grad * (scaling_num / scaling_den))
        
		value = param.get_value(borrow=True)
		# accu: accumulate gradient magnitudes
		accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
								broadcastable=param.broadcastable)
		# delta_accu: accumulate update magnitudes (recursively!)
		delta_accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
								broadcastable=param.broadcastable)

        # update accu (as in rmsprop)
		accu_new = rho * accu + (1 - rho) * grad ** 2
		updates[accu] = accu_new

		# compute parameter update, using the 'old' delta_accu
		update = (grad * T.sqrt(delta_accu + epsilon) /
					T.sqrt(accu_new + epsilon))
		updates[param] = param - learning_rate * update

		# update delta_accu (as accu, but accumulating updates)
		delta_accu_new = rho * delta_accu + (1 - rho) * update ** 2
		updates[delta_accu] = delta_accu_new

	return updates

def sgd_with_grad_clipping(loss_or_grads, params, learning_rate, rescale):

	grads = get_or_compute_grads(loss_or_grads, params)
	updates = OrderedDict()

	grad_norm = T.sqrt(sum(map(lambda x: T.sqr(x).sum(), grads)))
	not_finite = T.or_(T.isnan(grad_norm), T.isinf(grad_norm))
	grad_norm = T.sqrt(grad_norm)

	scaling_num = T.cast(rescale,dtype=theano.config.floatX)
	learning_rate = T.cast(learning_rate, dtype=theano.config.floatX)
	scaling_den = T.maximum(rescale, grad_norm)
	for param, grad in zip(params, grads):
		grad = T.switch(not_finite, 0.1 * param, grad * (scaling_num / scaling_den))
		updates[param] = param - learning_rate * grad

	return updates

def momentum_with_grad_clipping(loss_or_grads, params, learning_rate, momentum=0.9, rescale=1.0):

	updates = sgd_with_grad_clipping(loss_or_grads, params, learning_rate, rescale)
	return lasagne.updates.apply_momentum(updates, momentum=momentum)

def nesterov_momentum_with_grad_clipping(loss_or_grads, params, 
										learning_rate, momentum=0.9, rescale=1.0):

	updates = sgd_with_grad_clipping(loss_or_grads, params, learning_rate, rescale)
	return lasagne.updates.apply_nesterov_momentum(updates, momentum=momentum)

def batch_params_interruption(params, std_dev_weightNoise):
	updates = OrderedDict()
	for param in params:
		noise = theano_rng.normal(param.shape, 0, std_dev_weightNoise,dtype=theano.config.floatX)
		updates[param] = param + noise
	return updates
