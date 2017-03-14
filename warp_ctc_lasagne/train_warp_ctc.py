import numpy as np
import theano
import theano.tensor as T

import lasagne
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

# from theano.gpuarray import dnn
# from theano.gpuarray.type import gpuarray_shared_constructor
# from config import mode_with_gpu, mode_without_gpu, test_ctx_name, ref_cast
# from theano.configdefaults import SUPPORTED_DNN_CONV_ALGO_FWD
from six import StringIO

import gzip

from lasagne.layers import Layer
from lasagne.nonlinearities import identity, softmax, rectify, softplus, sigmoid, softmax, tanh
from lasagne import init
from utils import (f_logsumexp, log_softmax, epslog, 
					log_add, log_dot_matrix, 
					log_dot_tensor) 
#from markov_ctc import CTC
import clip_grad
import decode

from make_batches import (data_loader, iterate_batch_phoneme_data, 
						shuffle_batch_phoneme_data, iterate_batch_data)
from utils import *
import argparse
import decode
import pickle as pkl
from model import *
from lasagne.regularization import regularize_network_params, l2, l1
from gpu_ctc import ctc_gpu, ctc_gpu_costs

def main(filename, model, nonlinearitytype, blank_symbol, 
		main_model, dropout,
		train_scp, feats_scp, scp_len,
		spk2utt, pkl_data, label, optimizer, grad_clip,
		num_epochs, mini_batch, rescale, momentum,
		model_num, model_id, n_hidden, fineunte_epoch, grad_type, penalty, param_interput,
		eval_dev, load_model_flag, test_model, ctc_type,data_shuffle,
		dev_train_scp, dev_feats_scp, dev_scp_len, 
		dev_spk2utt, dev_pkl_data, dev_label,
		phoneme2int, phoneme_maps, output_phoneme, transform_phoneme):

	logger, rootpaths = init_logging(filename, nonlinearitytype, 0, 0)

	write_to_logger(logger, 'model_type '+ model)
	write_to_logger(logger, 'model config ')
	write_to_logger(logger, 'nonlinearitytype '+ nonlinearitytype)
	write_to_logger(logger, 'blank-symbol '+ str(blank_symbol))
	write_to_logger(logger, 'main_model '+str(main_model))
	write_to_logger(logger, 'dropout '+str(dropout))
	write_to_logger(logger, 'optimizer '+ str(optimizer))
	write_to_logger(logger, 'grad_clip '+ str(grad_clip))
	write_to_logger(logger, 'grad_type '+str(grad_type))
	write_to_logger(logger, 'penalty_type '+str(penalty))
	write_to_logger(logger, 'param_interput '+str(param_interput))

	write_to_logger(logger, 'epoch num '+ str(num_epochs))
	write_to_logger(logger, 'mini_batch size '+ str(mini_batch))
	write_to_logger(logger, 'rescale '+ str(rescale))
	write_to_logger(logger, 'momentum '+ str(momentum))
	write_to_logger(logger, 'model_num '+ str(model_num))
	write_to_logger(logger, 'model_id '+ str(model_id))
	write_to_logger(logger, 'n_hidden '+ str(n_hidden))
	write_to_logger(logger, 'fineunte_epoch '+ str(fineunte_epoch))

	write_to_logger(logger,'eval_type '+ eval_dev)
	write_to_logger(logger, 'load_model_flag '+ load_model_flag)
	write_to_logger(logger, 'test_model '+ test_model)
	write_to_logger(logger,'ctc_type '+ (ctc_type))
	write_to_logger(logger,'data_shuffle '+ (data_shuffle))
	
	write_to_logger(logger,'  train data config ')
	write_to_logger(logger,'train_scp_path '+ (train_scp))
	write_to_logger(logger,'feats_scp_path '+ (feats_scp))
	write_to_logger(logger,'scp_len_path '+ (scp_len))
	write_to_logger(logger,'spk2utt_path '+ (spk2utt))
	write_to_logger(logger,'pkl_data_path '+ (pkl_data))
	write_to_logger(logger,'label_path '+ (label))
	
	write_to_logger(logger,'  dev data config ')
	write_to_logger(logger,'dev_train_scp '+ (dev_train_scp))
	write_to_logger(logger,'dev_feats_scp '+ (dev_feats_scp))
	write_to_logger(logger,'dev_scp_len '+ (dev_scp_len))
	write_to_logger(logger,'dev_spk2utt '+ (dev_spk2utt))
	write_to_logger(logger,'dev_pkl_data '+ (dev_pkl_data))
	write_to_logger(logger,'dev_label '+ (dev_label))

	write_to_logger(logger, 'phoneme2int '+str(phoneme2int))
	write_to_logger(logger, 'phoneme_maps '+str(phoneme_maps))
	write_to_logger(logger, 'output_phoneme '+str(output_phoneme))
	write_to_logger(logger, 'transform_phoneme '+str(transform_phoneme))

	[truth_result_dict, 
	phoneme_dict, 
	truth_utt_phoneme_dict] = decode.read_output(dev_label, phoneme2int, output_phoneme)
	truth_transform_dict = decode.phoneme_transform(phoneme_maps, truth_utt_phoneme_dict, '61-to-39', transform_phoneme)

	total_label = len(phoneme_dict)+1
	write_to_logger(logger, 'label including blank '+str(total_label))

	print 'begin to load data'

	[sample_index, label_dict,
	label_mask, spk_seg_label, 
	ctc_label_dims, spk_nums] = data_loader(train_scp, feats_scp, scp_len,
												spk2utt, label)

	if main_model == 'build_warp_ctc_dropout_model':
		m_model = build_warp_ctc_dropout_model
		print 'building build dropout_model'

	if nonlinearitytype == 'tanh':
		nonlinearity = lasagne.nonlinearities.tanh
	elif nonlinearitytype == 'rectify':
		nonlinearity = lasagne.nonlinearities.rectify
	elif nonlinearitytype == 'sigmoid':
		nonlinearity = lasagne.nonlinearities.sigmoid
	elif nonlinearitytype == 'softplus':
		nonlinearity = lasagne.nonlinearities.softplus

	input_var = T.tensor3('input_var', dtype=theano.config.floatX)
	input_label = T.ivector('input_label')

	sym_lr = T.scalar('sym_lr',dtype=theano.config.floatX)
	sym_momentum = T.scalar('sym_momentum',dtype=theano.config.floatX)
	sym_rescale = T.scalar('sym_rescale',dtype=theano.config.floatX)

	sym_std = T.scalar()

	if grad_clip == 'not-clipping':
		grad_clipping = 0
	elif grad_clip == 'clipping':
		grad_clipping = 0
	transition_kernel_layer = m_model(input_var=input_var, input_dim=123, 
									n_hidden=int(n_hidden), grad_clip=grad_clipping,
									num_units=int(n_hidden), num_labels=(total_label), 
									nonlinearity_type=nonlinearity)

	if dropout == 'True':
		dropout_opt = False
	elif dropout == 'False':
		dropout_opt = True

	transition_kernel = lasagne.layers.get_output(transition_kernel_layer, deterministic=dropout_opt) # batch x seq x label_dim
	decpde_transition_kernel = lasagne.layers.get_output(transition_kernel_layer, deterministic=True) # batch x seq x label_dim
	log_probs = log_softmax(transition_kernel, axis=-1)

	if ctc_type == 'warp-ctc':
		input_lens = T.ivector('input_lens')
		label_lens = T.ivector('label_lens')

		cost = (ctc_gpu_costs(transition_kernel.dimshuffle((1,0,2)), input_lens, 
								input_label, label_lens)) # transition_kernel shuffule to seq x batch x label_dim

	all_params = lasagne.layers.get_all_params([transition_kernel_layer],trainable=True)
	for params in all_params:
		print params, params.get_value().shape

	if load_model_flag == 'load_model' and model_id is not None:
		print 'begin to load .pkl data'   
		load_model(model_id, filename, nonlinearitytype, 0, 0, all_params, model_num)
		pkl2mat(rootpaths, 0, 0, all_params)
		print 'succeeded in loading model parameters'

	if load_model_flag == 'load_mat_model' and model_id is not None: 
		print 'begin to load .mat data'
		load_mat_model(model_id, filename, nonlinearitytype ,0, 0, all_params, model_num)
		dump_model(rootpaths, 0, 0, all_params)
		print 'succeeded in loading model parameters'

	pkl2mat(rootpaths, 1, 1, all_params)
	dump_model(rootpaths, 1, 1, all_params)

	output_trace = T.argmax(decpde_transition_kernel, axis=-1) # since label_prob is batch x seq x label_dim

	if grad_type == 'warp-ctc':
		train_cost = T.mean(cost)

	if penalty == 'l2':
		print '--------------using l2_penalty----------'
		l2_penalty = regularize_network_params(transition_kernel_layer, l2) * 1e-4
		train_cost += l2_penalty
	elif penalty == 'l1':
		print '--------------using l1_penalty-----------'
		l1_penalty = regularize_network_params(transition_kernel_layer, l1) * 1e-4
		train_cost += l1_penalty
	elif penalty == 'l1-l2':
		print '--------------using l2 and l1 penalty-------------'
		l1_penalty = regularize_network_params(transition_kernel_layer, l1) * 1e-4
		l2_penalty = regularize_network_params(transition_kernel_layer, l2) * 1e-4
		train_cost = train_cost + l2_penalty + l1_penalty

	grads = T.grad(train_cost, all_params, disconnected_inputs='warn')
	
	grad_norm = T.sqrt(sum(map(lambda x: T.sqr(x).sum(), grads)))
	param_norm = T.sqrt(sum(map(lambda x: T.sqr(x).sum(), all_params)))
	
	if grad_clip == 'not-clipping':
		updates_sgd = lasagne.updates.sgd(grads, all_params, learning_rate=sym_lr)

		updates_nesterov = lasagne.updates.nesterov_momentum(grads, all_params, 
											learning_rate=sym_lr, momentum=sym_momentum)

		updates_adam = lasagne.updates.adam(grads, all_params, learning_rate=sym_lr, beta1=0.9,
												beta2=0.999, epsilon=1e-8)

		updates_adadelta = lasagne.updates.adadelta(grads, all_params, learning_rate=sym_lr, 
									rho=0.95, epsilon=1e-6)

		updates_momentum = lasagne.updates.momentum(grads, all_params, learning_rate=sym_lr, 
												momentum=sym_momentum)

		updates_rmsprop = lasagne.updates.rmsprop(grads, all_params, learning_rate=sym_lr,
															rho=0.9, epsilon=1e-6)

		updates_finetune = lasagne.updates.sgd(grads, all_params, learning_rate=sym_lr)

	elif grad_clip == 'clipping':
		updates_adam = clip_grad.adam_with_grad_clipping(grads, all_params, learning_rate=sym_lr, beta1=0.9,
												beta2=0.999, epsilon=1e-8, rescale=sym_rescale)

		updates_momentum = clip_grad.momentum_with_grad_clipping(grads, all_params, learning_rate=sym_lr, 
												momentum=sym_momentum, rescale=sym_rescale)

		updates_adadelta = clip_grad.adadelta_with_grad_clipping(grads, all_params, 
									learning_rate=sym_lr, rho=0.95, epsilon=1e-6, rescale=sym_rescale)

		updates_rmsprop = clip_grad.rmsprop_with_grad_clipping(grads, all_params, learning_rate=sym_lr,
															rho=0.9, epsilon=1e-6, rescale=sym_rescale)

		updates_finetune = clip_grad.sgd_with_grad_clipping(grads, all_params, learning_rate=sym_lr, rescale=sym_rescale)

		updates_nesterov = clip_grad.nesterov_momentum_with_grad_clipping(grads, all_params, learning_rate=sym_lr, 
												momentum=sym_momentum, rescale=sym_rescale)
		updates_sgd = clip_grad.sgd_with_grad_clipping(grads, all_params, learning_rate=sym_lr, 
														rescale=sym_rescale)

	if param_interput == 'gaussian':
		print '-----------interput params with gaussian noise----------'
		updates = clip_grad.batch_params_interruption(all_params, sym_std)
		interput = theano.function([sym_std],[],updates=updates)

	if optimizer == 'adam':
		updates = updates_adam
		lr = 1e-4
	elif optimizer == 'momentum_sgd':
		updates = updates_momentum
		lr = 1e-4
	elif optimizer == 'rmsprop':
		updates = updates_rmsprop
		lr = 1e-4
	elif optimizer == 'adadelta':
		updates = updates_adadelta
		lr = 1e-4
	elif optimizer == 'nesterov_momentum':
		updates = updates_nesterov
		lr = 1e-4
	elif optimizer == 'sgd':
		updates = updates_sgd
		lr = 1e-4

	write_to_logger(logger, 'initial_lr '+str(lr))

	if ctc_type == 'warp-ctc':
		inputs = [input_var, input_label, input_lens, label_lens, sym_lr, sym_rescale, sym_momentum, sym_std]
		outputs = [cost, train_cost, output_trace, grad_norm, param_norm, log_probs, transition_kernel]
	
	print '-----------begin to monitor grads of training------------'
	monitor_fn = theano.function(inputs, [grad_norm]+grads, 
								allow_input_downcast=True,
								on_unused_input='warn')

	print "---begin to compling training function---"
	tain_fn = theano.function(inputs, outputs, 
								updates=updates, 
								allow_input_downcast=True,
								on_unused_input='warn')

	print "---begin to compling finetune function---"
	finetune_fn = theano.function(inputs, outputs, 
								updates=updates_finetune, 
								allow_input_downcast=True,
								on_unused_input='warn')

	print "---begin to compling predict function---"
	predict = theano.function(inputs, outputs,
								allow_input_downcast=True,
								on_unused_input='warn')

	# Finally, launch the training loop.
	train_output_label_path = get_socres_path(rootpaths, 'train', 61, 61)
	
	print("Starting training...")

	if test_model =='train':
		print '---------begin to train model----------'
		epochs = num_epochs
		function_api = tain_fn
	elif test_model == 'eval':
		print '---------begin to eval model------------'
		epochs = 1
		train_fwrite = open(train_output_label_path,'w')
		function_api = predict
	elif test_model == 'finetune':
		function_api = finetune_fn
		epochs = num_epochs
		if param_interput == 'gaussian':
			interput(lasagne.utils.floatX(0.075))

	itera_num = 0
	grads_check = 'non-check'
	only_one_data = 'False'
	load_data_flag = 1
	old_cost = 100
	cnt = 0
	pkl_path = '/mnt/aa200d57-8c3d-48ef-a84e-116572f32d45/ctc/ctc1.pkl'
	my_pkl_path = '/mnt/aa200d57-8c3d-48ef-a84e-116572f32d45/ctc/my-ctc.pkl'

	for epoch in range(epochs):
		write_to_logger(logger, 'lr '+str(lr))
		lr_floatx = lasagne.utils.floatX(lr)
		rescale_floatx = lasagne.utils.floatX(rescale)
		momentum_floatx = lasagne.utils.floatX(momentum)
		std_floatx = lasagne.utils.floatX(np.power(lr, 0.55))
		cnt = 0
		data_num = 0
		start_time_all = time.time()
		start_time = time.time()
		batch_num = 0
		total_cost = 0

		if np.mod(epoch,1) == 0:
			name = 'dev_'+str(epoch)
			dev_output_label_path = get_socres_path(rootpaths, name, 61, 61)
			dev_fwrite = open(dev_output_label_path,'w')

			batch_per = 0
			dev_num = 0
			batch_ctc = 0
			[dev_sample_index, dev_label_dict,
			dev_label_mask, dev_spk_seg_label, 
			dev_ctc_label_dims, dev_spk_nums] = data_loader(dev_train_scp, 
														dev_feats_scp, 
														dev_scp_len,
														dev_spk2utt, 
														dev_label)

			for batch in iterate_batch_data(dev_pkl_data, gzip.open, dev_sample_index, 
										dev_label_dict, dev_label_mask, dev_spk_seg_label, dev_spk_nums, mini_batch):
				[batch_data, batch_label, batch_label_len, batch_data_len, 
					batch_one_hot_label, batch_seg_label, content_name] = batch

				[cost, 
				train_cost,
				output_trace, 
				grad_norm,
				param_norm,
				log_probs,
				transition_kernel] = predict(batch_data,
										batch_label,
										batch_data_len,
										batch_label_len,
			 							lr_floatx, 
			 							rescale_floatx,
			 							momentum_floatx,
			 							std_floatx)
				dev_num += batch_data.shape[0]
				batch_ctc = train_cost*batch_data.shape[0] + batch_ctc

				for k in xrange(batch_data.shape[0]):
					dev_fwrite.write(content_name[k])
					dev_fwrite.write(' ')
					for j in output_trace[k]:
						dev_fwrite.write(str(j))
						dev_fwrite.write(' ')
					dev_fwrite.write('\n')

			write_to_logger(logger, 'dev-ctc-cost '+str(batch_ctc/dev_num)+' dev total utterance '+str(dev_num))

			dev_fwrite.close()

			[dev_result_dict, 
			phoneme_dict, 
			dev_utt_phoneme_dict] = decode.read_output(dev_output_label_path, phoneme2int, output_phoneme)
			dev_transform_dict = decode.phoneme_transform(phoneme_maps, dev_utt_phoneme_dict, '61-to-39', transform_phoneme)

			score = []
			num_phoneme = 0
			for key in dev_transform_dict.keys():
				tmp_score = decode.calcPER(truth_transform_dict[key], dev_transform_dict[key])
				num_phoneme += len(truth_transform_dict[key])
				score.append(tmp_score)

			score = np.asarray(score)
			per = np.sum(score) / num_phoneme
			write_to_logger(logger, 'finial '+str(epoch)+' dev ctc-cost '+str(batch_ctc/dev_num)+' dev-per '+str(per))

		if data_shuffle == 'based-on-lens':
			all_data = iterate_batch_data(pkl_data, gzip.open, sample_index, 
											label_dict, label_mask, spk_seg_label, spk_nums, mini_batch)
		

		if only_one_data == 'True' and load_data_flag == 1:
			train_k = -1
			for batch in all_data:
				[batch_data, batch_label, batch_label_len, batch_data_len, 
					batch_one_hot_label, batch_seg_label, content_name] = batch
				if batch_num[0] == 'mpar0_si1576':
					print batch_data.shape, content_name, batch_data_len
					train_k = k
					break
			load_data_flag = 0
		if load_data_flag == 0:
			cnt += 1
			[cost, 
			train_cost,
			output_trace, 
			grad_norm,
			param_norm,
			log_probs,
			transition_kernel] = function_api(batch_data,
											batch_label,
											batch_data_len,
											batch_label_len,
			 								lr_floatx, 
			 								rescale_floatx,
			 								momentum_floatx,
			 								std_floatx)

			if grads_check == 'check':
				grads = monitor_fn(batch_data,
								batch_label,
								batch_data_len,
								batch_label_len,
			 					lr_floatx, 
			 					rescale_floatx,
			 					momentum_floatx,
			 					std_floatx)

			print lr
			print train_cost
			print content_name
			print output_trace
			print batch_label
			print batch_label_len
			print train_k
			print batch_label.shape
			print np.sum(batch_label_len)
			write_to_logger(logger, 'ctc-cost '+str(train_cost)+' grad-norm '+str(grad_norm)+' param-norm '+str(param_norm))
			pkl.dump([cost,output_trace,
						transition_kernel,log_probs,
						batch_data,batch_label, 
						batch_data_len, batch_label_len,
						content_name], open(my_pkl_path, "wb"), protocol=pkl.HIGHEST_PROTOCOL)
		if only_one_data == 'False':
			if data_shuffle == 'based-on-lens':
				for batch in all_data:
					[batch_data, batch_label, batch_label_len, batch_data_len, 
					batch_one_hot_label, batch_seg_label, content_name] = batch
			
					batch_data_num = batch_data.shape[0]
					data_num += batch_data_num

					[cost, 
					train_cost,
					output_trace, 
					grad_norm,
					param_norm,
					log_probs,
					transition_kernel] = function_api(batch_data,
											batch_label,
											batch_data_len,
											batch_label_len,
			 								lr_floatx, 
			 								rescale_floatx,
			 								momentum_floatx,
			 								std_floatx)

					if test_model == 'eval':
						for k in xrange(batch_data.shape[0]):
							train_fwrite.write(content_name[k])
							train_fwrite.write(' ')
							for j in output_trace[k]:
								train_fwrite.write(str(j))
								train_fwrite.write(' ')
							train_fwrite.write('\n')

					# pkl.dump([train_cost,output_trace,
					# 	transition_kernel,
					# 	batch_data,batch_label, log_probs,
					# 	batch_data_len, batch_label_len,
					# 	content_name], open(pkl_path, "wb"), protocol=pkl.HIGHEST_PROTOCOL)

					batch_num += 1
					total_cost = train_cost*batch_data_num + total_cost
					if batch_num <= 20 and batch_num >= 0:
						end_time = time.time()
						print 'ctc cost ', train_cost, ' with timimg ', end_time - start_time
					start_time = time.time()
					#write_to_logger(logger, ' batch_data_num '+str(batch_data_num)+' ctc-cost '+str(train_cost)+' grad_norm '+str(grad_norm)+' param-norm '+str(param_norm))

					cnt += 1

					if np.mod(cnt,50) == 0:
						end_time = time.time()
						print 'sentence ', content_name
						print 'decoding ', output_trace
						print 'ground-turth ', batch_label
						print 'label len ', batch_label_len
						print 'seq len ', batch_data_len
						print cnt, '-th iterations with ',end_time - start_time, ' time consuming'
						start_time = time.time()

			if test_model == 'eval':
				train_fwrite.close()

			p = get_model_mat_path(rootpaths, 0, 0)
			pkl2mat(rootpaths, epoch, epoch, all_params)
			dump_model(rootpaths, epoch, epoch, all_params)

			end_time_all = time.time()

			total_cost /= data_num

			abs_improvment = np.abs(old_cost - total_cost) / old_cost
			write_to_logger(logger, 'relative improvement '+str(abs_improvment))
			old_cost = total_cost

			if epoch >= 50 and abs_improvment <= 1e-2:
				if param_interput == 'gaussian':
					interput(std_floatx)
				function_api = finetune_fn
				write_to_logger(logger, '--------------------------begin to to finetune using sgd-----------------')
				if lr >= 1e-8:
					lr *= 0.5
				if momentum >= 0.5:
					momentum *= 0.9

			write_to_logger(logger, 'epoch '+str(epoch)+' time '+str(end_time_all - start_time_all)+' ctc-cost '+str(total_cost))

	if eval_dev == 'PER':

		dev_output_label_path = get_socres_path(rootpaths, 'dev', 61, 61)
		dev_fwrite = open(dev_output_label_path,'w')

		lr_floatx = lasagne.utils.floatX(lr)
		rescale_floatx = lasagne.utils.floatX(rescale)
		momentum_floatx = lasagne.utils.floatX(momentum)
		std_floatx = lasagne.utils.floatX(np.power(lr, 0.55))

		batch_per = 0
		dev_num = 0
		batch_ctc = 0
		[dev_sample_index, dev_label_dict,
		dev_label_mask, dev_spk_seg_label, 
		dev_ctc_label_dims, dev_spk_nums] = data_loader(dev_train_scp, 
														dev_feats_scp, 
														dev_scp_len,
														dev_spk2utt, 
														dev_label)

		for batch in iterate_batch_data(dev_pkl_data, gzip.open, dev_sample_index, 
										dev_label_dict, dev_label_mask, dev_spk_seg_label, dev_spk_nums, mini_batch):
			[batch_data, batch_label, batch_label_len, batch_data_len, 
					batch_one_hot_label, batch_seg_label, content_name] = batch

			[cost, 
			train_cost,
			output_trace, 
			grad_norm,
			param_norm,
			log_probs,
			transition_kernel] = predict(batch_data,
										batch_label,
										batch_data_len,
										batch_label_len,
			 							lr_floatx, 
			 							rescale_floatx,
			 							momentum_floatx,
			 							std_floatx)
			dev_num += batch_data.shape[0]
			batch_ctc = train_cost*batch_data.shape[0] + batch_ctc

			for k in xrange(batch_data.shape[0]):
				dev_fwrite.write(content_name[k])
				dev_fwrite.write(' ')
				for j in output_trace[k]:
					dev_fwrite.write(str(j))
					dev_fwrite.write(' ')
				dev_fwrite.write('\n')

		write_to_logger(logger, 'ctc-cost '+str(batch_ctc/dev_num)+' total utterance '+str(dev_num))

		dev_fwrite.close()

		[dev_result_dict, 
		phoneme_dict, 
		dev_utt_phoneme_dict] = decode.read_output(dev_output_label_path, phoneme2int, output_phoneme)
		dev_transform_dict = decode.phoneme_transform(phoneme_maps, dev_utt_phoneme_dict, '61-to-39', transform_phoneme)

		score = []
		num_phoneme = 0
		for key in dev_transform_dict.keys():
			tmp_score = decode.calcPER(truth_transform_dict[key], dev_transform_dict[key])
			num_phoneme += len(truth_transform_dict[key])
			score.append(tmp_score)

		score = np.asarray(score)
		per = np.sum(score)/num_phoneme
		write_to_logger(logger, 'finial '+str(epoch)+' dev ctc-cost '+str(batch_ctc/dev_num)+' dev-per '+str(per))

		search_path = '/root/warp_ctc_search_info.txt'
		search_fwrite = open(search_path,'a')
		lines = 'dev ctc-cost '+ str(total_cost)+ ' test ctc-cost '+str(batch_ctc/dev_num)
		search_fwrite.write(lines)
		search_fwrite.write('\n')

def parse_args():
	"""parse input arguments"""
	parser = argparse.ArgumentParser(description='markov_ctc')
	parser.add_argument('--filename',dest='filename',help="output file destination",
						default='markov-ctc')
	parser.add_argument('--model',dest='model_type',help="output file destination",
						default='markov-ctc')
	parser.add_argument('--nonlinearitytype',dest='nonlinearitytype',help="nonlinearitytype",
						default='sigmoid')
	parser.add_argument('--blank_symbol',dest='blank_symbol',help="blank_symbol", default='0')
	parser.add_argument('--main_model', dest='main_model', help='main_model')
	parser.add_argument('--dropout', dest='dropout', help='dropout')

	parser.add_argument('--train_scp',dest='train_scp',help="train_scp")
	parser.add_argument('--feats_scp',dest='feats_scp',help="feats_scp")
	parser.add_argument('--scp_len',dest='scp_len',help="scp_len")
	parser.add_argument('--spk2utt',dest='spk2utt',help="spk2utt")
	parser.add_argument('--pkl_data',dest='pkl_data',help="pkl_data")
	parser.add_argument('--label',dest='label',help="label")
	parser.add_argument('--optimizer',dest='optimizer',help="optimizer")
	parser.add_argument('--grad_clip',dest='grad_clip',help="grad_clip")

	parser.add_argument('--epoch',dest='epoch',help="epoch")
	parser.add_argument('--mini_batch',dest='mini_batch',help="mini_batch")
	parser.add_argument('--rescale',dest='rescale',help="rescale",default='1.0')
	parser.add_argument('--momentum',dest='momentum',help="rescale",default='0.9')
	parser.add_argument('--model_num',dest='model_num',help="model_num")
	parser.add_argument('--model_id',dest='model_id',help="load to tune when load_model=1 ...\
						or end-to-end train for load_model=0 ")
	parser.add_argument('--n_hidden',dest='n_hidden',help="n_hidden")
	parser.add_argument('--fineunte_epoch',dest='fineunte_epoch',help="fineunte_epoch")
	parser.add_argument('--grad_type',dest='grad_type',help="grad_type")
	parser.add_argument('--penalty_type',dest='penalty_type',help="penalty_type")
	parser.add_argument('--param_interput', dest='param_interput', help='param_interput')

	parser.add_argument('--eval_dev',dest='eval_dev',help="eval_dev")
	parser.add_argument('--load_model_flag',dest='load_model_flag',help="load_model_flag",
						default='load_model')
	parser.add_argument('--test_model',dest='test_model',help="load to tune when load_model=1 ...\
						or end-to-end train for load_model=0 ",default='train')
	parser.add_argument('--ctc_type',dest='ctc_type',help="log/plain-markov or standard ctc ",
						default='markov-plain')
	parser.add_argument('--data_shuffle',dest='data_shuffle',help="log/plain-markov or standard ctc ",
						default='data_shuffle')
	parser.add_argument('--dev_train_scp',dest='dev_train_scp',help="dev_train_scp")
	parser.add_argument('--dev_feats_scp',dest='dev_feats_scp',help="dev_feats_scp")
	parser.add_argument('--dev_scp_len',dest='dev_scp_len',help="dev_scp_len")
	parser.add_argument('--dev_spk2utt',dest='dev_spk2utt',help="dev_spk2utt")
	parser.add_argument('--dev_pkl_data',dest='dev_pkl_data',help="dev_pkl_data")
	parser.add_argument('--dev_label',dest='dev_label',help="dev_label")

	parser.add_argument('--phoneme2int',dest='phoneme2int',help="phoneme2int")
	parser.add_argument('--phoneme_maps',dest='phoneme_maps',help="phoneme_maps")
	parser.add_argument('--output_phoneme',dest='output_phoneme',help="output_phoneme")
	parser.add_argument('--transform_phoneme',dest='transform_phoneme',help="transform_phoneme")

	args = parser.parse_args()
	return args

if __name__ == '__main__':

	args = parse_args()

	filename = args.filename
	model = args.model_type
	nonlinearitytype = args.nonlinearitytype
	blank_symbol = args.blank_symbol
	main_model = args.main_model
	dropout = args.dropout

	train_scp = args.train_scp
	feats_scp = args.feats_scp
	scp_len = args.scp_len
	spk2utt = args.spk2utt
	pkl_data = args.pkl_data
	label = args.label
	optimizer = args.optimizer
	grad_clip = args.grad_clip

	epoch = int(args.epoch)
	mini_batch = int(args.mini_batch)
	rescale = float(args.rescale)
	momentum = float(args.momentum)
	model_num = int(args.model_num)
	model_id = int(args.model_id)
	n_hidden = int(args.n_hidden)
	fineunte_epoch = int(args.fineunte_epoch)
	grad_type = args.grad_type
	penalty_type = args.penalty_type
	param_interput = args.param_interput

	eval_dev = args.eval_dev
	load_model_flag = args.load_model_flag
	test_model = args.test_model
	ctc_type = args.ctc_type
	data_shuffle = args.data_shuffle

	dev_train_scp = args.dev_train_scp
	dev_feats_scp = args.dev_feats_scp
	dev_scp_len = args.dev_scp_len
	dev_spk2utt = args.dev_spk2utt
	dev_pkl_data = args.dev_pkl_data
	dev_label = args.dev_label

	phoneme2int = args.phoneme2int
	phoneme_maps = args.phoneme_maps
	output_phoneme = args.output_phoneme
	transform_phoneme = args.transform_phoneme

	main(filename=filename, model=model, nonlinearitytype=nonlinearitytype, 
		blank_symbol=blank_symbol, main_model=main_model, dropout=dropout,
		train_scp=train_scp, feats_scp=feats_scp, scp_len=scp_len,
		spk2utt=spk2utt, pkl_data=pkl_data, label=label, optimizer=optimizer, grad_clip=grad_clip,
		num_epochs=epoch, mini_batch=mini_batch, rescale=rescale, momentum=momentum,
		model_num=model_num, model_id=model_id, n_hidden=n_hidden, fineunte_epoch=fineunte_epoch, 
		grad_type=grad_type, penalty=penalty_type, param_interput=param_interput,
		eval_dev=eval_dev, load_model_flag=load_model_flag, 
		test_model=test_model, ctc_type=ctc_type, data_shuffle=data_shuffle,
		dev_train_scp=dev_train_scp, dev_feats_scp=dev_feats_scp, dev_scp_len=dev_scp_len, 
		dev_spk2utt=dev_spk2utt, dev_pkl_data=dev_pkl_data, dev_label=dev_label,
		phoneme2int=phoneme2int, phoneme_maps=phoneme_maps, 
		output_phoneme=output_phoneme, 
		transform_phoneme=transform_phoneme)


