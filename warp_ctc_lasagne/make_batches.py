import theano

import gzip
import cPickle as pkl
import sys
import numpy as np
from itertools import izip
import random
import argparse

from itertools import izip
from collections import OrderedDict
import time
import scipy.io as sio

def time2string(time_string):
	string_time = ''
	for k in xrange(5,0,-1):
		basis = np.power(10,k)
		s = int(time_string/basis)
		string_time += str(s)
		if s > 0:
			time_string -= (s*basis)
	string_time += str(time_string)
	return string_time

def one_hot(labels, total_classes):
	'''
	Converts an array of label integers to a one-hot matrix encoding
	:parameters:
		- labels : np.ndarray, dtype=int
			Array of integer labels, in {0, n_classes - 1}
		- n_classes : int
			Total number of classes
	:returns:
		- one_hot : np.ndarray, dtype=bool, shape=(labels.shape[0], n_classes)
			One-hot matrix of the input
	'''
	one_hot_coding = np.zeros((labels.shape[0], total_classes)).astype('int32')
	for k in xrange(labels.shape[0]):
		one_hot_coding[k,labels[k]] = 1

	return one_hot_coding  

def stream_file(filename,open_method=gzip.open):
	with open_method(filename,'rb') as fd:
		try:
			while True:
				x = pkl.load(fd)
				yield x
		except EOFError: pass

def lenscp2trainIndex(feat_scp, len_temp, train_scp):
	try:
		frobj_scp = open(feat_scp,'r')
		frobj_len = open(len_temp,'r')
		frobj_train_scp = open(train_scp,'r')
	except IOError:
		print "failed to read ", feat_scp
		print "failed to read ", len_temp
		print "failed to read ", train_scp
	else:
		scp = frobj_scp.readlines()
		len_scp = frobj_len.readlines()
		train_scp = frobj_train_scp.readlines()
		sample_index = OrderedDict()
		feats_index = OrderedDict()
		for scp_line, len_line in zip(scp,len_scp):
			content = scp_line.strip().split()
			lens = int(len_line.strip())
			name = content[0]
			feats_index[name] = lens
		for scp_line in train_scp:
			content = scp_line.strip().split()
			name = content[0]
			if sample_index.has_key(feats_index[name]) is False:
				sample_index[feats_index[name]] = [name]
			else:
				sample_index[feats_index[name]].append(name)
		frobj_scp.close()
		frobj_len.close()
		frobj_train_scp.close()

		return sample_index

def spklabel(spk2utt):
	try:
		frobj = open(spk2utt,'r')
	except IOError:
		print "failed to read ", spk2utt
	else:
		utt = frobj.readlines()
		spk_seg_label = OrderedDict()
		spk_label_ = OrderedDict()
		spkcnt = 0
		for eachline in utt:
			content = eachline.strip().split()
			spk = content[0]
			spk_segs = content[1:]
			spk_label_[spk] = spkcnt
			for spk_seg in spk_segs:
				spk_seg_label[spk_seg] = spk_label_[spk]
			spkcnt += 1
		frobj.close()
		num_spks = len(spk_label_)
		print 'numbers of speakers are ', num_spks

		return spk_seg_label, num_spks

def read_spk2utt(spk2utt):
	try:
		frobj = open(spk2utt,'r')
	except IOError:
		print "failed to read ", spk2utt
	else:
		utt = frobj.readlines()
		spk2utt_dict = OrderedDict()
		spkindex = []
		cnt = 0
		for eachline in utt:
			content = eachline.strip().split()
			spk = content[0]
			spk_segs = content[1:]
			utt_num = len(spk_segs)
			spkindex.append([cnt, cnt+utt_num])
			cnt += utt_num
			for spk_seg in spk_segs:
				if spk2utt_dict.has_key(spk):
					spk2utt_dict[spk].append(spk_seg)
				else:
					spk2utt_dict[spk] = [spk_seg]
		frobj.close()
		print spkindex[-1]

	return spk2utt_dict, np.array(spkindex).astype( 'int32' )

def read_train_scp(train_scp):
	try:
		frobj_train_scp = open(train_scp,'r')
	except IOError:
		print "failed to read ", train_scp
	else:
		train_scp = frobj_train_scp.readlines()
		seg_dict = OrderedDict()
		for scp_line in train_scp:
			content = scp_line.strip().split()
			vad_name = content[0]
			utt_name = vad_name.split('_')[0]
			if seg_dict.has_key(utt_name):
				seg_dict[utt_name].append(vad_name)
			else:
				seg_dict[utt_name] = [vad_name]
	return seg_dict

def readlabel(seg_phone_label):
	#<unk> 1
	#[laughter] 3
	#[noise] 2
	#[vocalized-noise] 1
	total_label = []
	try:
		frobj = open(seg_phone_label,'r')
	except IOError:
		print "failed to read ", seg_phone_label
	else:
		label = frobj.readlines()
		label_dict = OrderedDict()
		label_mask = OrderedDict()
		cnt = 0 
		for eachline in label:
			content = eachline.strip().split()
			name = content[0]
			if cnt == 0:
				print name
			cnt += 1
			label_dict[name] = []
			for l in content[1:]:
				t = int(l)
				total_label.append(t)
				# if t == 28:
				#  	continue
				label_dict[name].append(t)
			if len(content[1:]) == 1:
				if int(content[1:][0]) == 1 or int(content[1:][0]) == 2 or int(content[1:][0]) == 3:
					label_mask[name] = 0
				else:
					label_mask[name] = 1
			else:
				label_mask[name] = 1
		frobj.close()
		classes, labels = np.unique(total_label, return_inverse=True)
		print 'phoneme characters ', classes, ' without ctc-blanks ', len(classes)
		ctc_label_classes = len(classes) + 1
		print 'numbers of ctc label ', ctc_label_classes
		return label_dict, label_mask, ctc_label_classes

def read_unit2state(unit2state):
	try:
		frobj = open(unit2state,'r') 
	except IOError:
		print "failed to read ", unit2state
	else:
		info = frobj.readlines()
		unit2state_dict = OrderedDict()
		for eachline in info:
			content = eachline.strip().split()
			l = []
			for item in content[1:]:
				l.append(int(item))
			unit2state_dict[content[0]] = l
		frobj.close()
		return unit2state_dict

def data_loader(train_scp, feats_scp, scp_len,
				spk2utt, label):

	spk_seg_label, num_spks = spklabel(spk2utt)
	sample_index = lenscp2trainIndex(feats_scp, scp_len, train_scp)
	
	label_dict, label_mask, ctc_label_dims = readlabel(label)

	return sample_index, label_dict, label_mask, spk_seg_label, ctc_label_dims, num_spks

def data_state_loader(train_scp, feats_scp, scp_len,
				spk2utt, label, unit2state):

	spk_seg_label, num_spks = spklabel(spk2utt)
	sample_index = lenscp2trainIndex(feats_scp, scp_len, train_scp)
	
	label_dict, label_mask, ctc_label_dims = readlabel(label)
	unit2state_dict = read_unit2state(unit2state)

	return [sample_index, label_dict, label_mask, 
	spk_seg_label, ctc_label_dims, num_spks, unit2state_dict]


def label2mask(label_list, label_len, blank_symbol):
	# for current, blank_symbol better to set 0 with consistency og essen
	assert len(label_list) == len(label_len)
	label_len = np.array(label_len)
	max_len = np.max(label_len)
	batch_label = np.zeros((label_len.shape[0],max_len)).astype('int32')
	batch_mask = np.zeros((label_len.shape[0],max_len)).astype('int32')
	for t in xrange(len(label_list)):
		batch_label[t,0:label_len[t]] = label_list[t]
		batch_label[t,label_len[t]:max_len] = blank_symbol
		batch_mask[t,0:label_len[t]] = 1

	return batch_label, batch_mask

def label2state(state_list, state_len, unit_dict, blank_symbol):
	assert len(state_list) == len(state_len)
	label_len = np.array(state_len)
	max_len = np.max(label_len)
	batch_label = np.zeros((label_len.shape[0],max_len)).astype('int32')
	batch_mask = np.zeros((label_len.shape[0],max_len)).astype('int32')
	if blank_symbol == 'eps':
		blank_state = unit_dict[blank_symbol]
		blank_state_len = len(blank_state)
	for t in xrange(len(state_list)):
		batch_label[t,0:label_len[t]] = state_list[t]
		#for k in range(state_len):
		batch_label[t,label_len[t]:max_len] = 0
		batch_mask[t,0:label_len[t]] = 1
	return batch_label, batch_mask

def extract_mvn(pkl_data, open_method, pkl_normalize_info):
	cnt = 0
	with open_method(pkl_data,'rb') as fd:
		try:
			while True:
				x = pkl.load(fd)
				if cnt == 0:
					data_matrix = x[1]
				else:
					data_matrix = np.vstack([data_matrix,x[1]])
		except EOFError: pass
		data_matrix = np.asarray(data_matrix).astype('float32')
		mean_vector = np.mean(data_matrix, axis=0, keepdims=True)
		std_var = np.std(data_matrix, axis=0, keepdims=True)
	pkl.dump([mean_vector, std_var], open(pkl_normalize_info, "wb"), protocol=pkl.HIGHEST_PROTOCOL)

	return data_matrix, mean_vector, std_var

def shuffle_batch_phoneme_data(pkl_data, open_method, sample_index, 
								mini_batch, label_dict, blank_symbol):
	streams = stream_file(pkl_data,open_method)
	batch_data = []
	for key in sample_index.keys():

		batch = len(sample_index[key])

		batch_size = 0
		while batch_size + mini_batch <= batch:
			content_name = []
			cnt = 0
			label = []
			label_len = []
			start_n = batch_size
			end_n = batch_size + mini_batch

			batch_size = end_n
			for k in xrange(mini_batch):
				content = streams.next()
				if cnt == 0:
					n_frame = content[1].shape[0]
					n_dim = content[1].shape[1]
					data = np.zeros((mini_batch, n_frame, n_dim))
				content_name.append(content[0])
				cnt += 1
				data[k,:,:] = content[1] # load data with (batches, lens, dims)
				label.append(label_dict[content[0]])
				label_len.append(len(label_dict[content[0]]))
			batch_label, batch_mask = label2mask(label, label_len, blank_symbol)
			batch_data.append([np.array(data).astype('float32'), batch_label,batch_mask,content_name])
		
		if batch_size < batch:
			cnt = 0
			content_name = []
			label = []
			label_len = []
			left_index = xrange(batch_size, batch)

			for k in xrange(len(left_index)):
				content = streams.next()
				if cnt == 0:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
					n_frame = content[1].shape[0]
					n_dim = content[1].shape[1]
					data = np.zeros((len(left_index), n_frame, n_dim))
				content_name.append(content[0])
				cnt += 1
				data[k,:,:] = content[1] # load data with (batches, lens, dims)
				label.append(label_dict[content[0]])
				label_len.append(len(label_dict[content[0]]))
			batch_label, batch_mask = label2mask(label, label_len, blank_symbol)
			batch_data.append([np.array(data).astype('float32'), batch_label,batch_mask,content_name])

	return batch_data

def iterate_batch_phoneme_normalized_data(pkl_data, open_method, sample_index, 
	mini_batch, label_dict, blank_symbol, train_mean, train_std):
	streams = stream_file(pkl_data,open_method)
	for key in sample_index.keys():

		batch = len(sample_index[key])

		batch_size = 0
		while batch_size + mini_batch <= batch:
			content_name = []
			cnt = 0
			label = []
			label_len = []
			start_n = batch_size
			end_n = batch_size + mini_batch

			batch_size = end_n
			for k in xrange(mini_batch):
				content = streams.next()
				if cnt == 0:
					n_frame = content[1].shape[0]
					n_dim = content[1].shape[1]
					data = np.zeros((mini_batch, n_frame, n_dim))
				content_name.append(content[0])
				cnt += 1
				data[k,:,:] = content[1] # load data with (batches, lens, dims)
				label.append(label_dict[content[0]])
				label_len.append(len(label_dict[content[0]]))
			batch_label, batch_mask = label2mask(label, label_len, blank_symbol)
			data = np.array(data).astype('float32')
			data = (data - train_mean) / train_std
			yield data, batch_label, batch_mask, content_name
		
		if batch_size < batch:
			cnt = 0
			content_name = []
			label = []
			label_len = []
			left_index = xrange(batch_size, batch)

			for k in xrange(len(left_index)):
				content = streams.next()
				if cnt == 0:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
					n_frame = content[1].shape[0]
					n_dim = content[1].shape[1]
					data = np.zeros((len(left_index), n_frame, n_dim))
				content_name.append(content[0])
				cnt += 1
				data[k,:,:] = content[1] # load data with (batches, lens, dims)
				label.append(label_dict[content[0]])
				label_len.append(len(label_dict[content[0]]))
			batch_label, batch_mask = label2mask(label, label_len, blank_symbol)
			data = np.array(data).astype('float32')
			data = (data - train_mean) / train_std
			yield data, batch_label, batch_mask, content_name	

def iterate_batch_state_phoneme_data(pkl_data, open_method, sample_index, 
								mini_batch, label_dict, blank_symbol, unit2state_dict):
	streams = stream_file(pkl_data,open_method)
	for key in sample_index.keys():

		batch = len(sample_index[key])

		batch_size = 0
		while batch_size + mini_batch <= batch:
			content_name = []
			cnt = 0
			label = []
			label_len = []
			start_n = batch_size
			end_n = batch_size + mini_batch

			batch_size = end_n
			for k in xrange(mini_batch):
				content = streams.next()
				if cnt == 0:
					n_frame = content[1].shape[0]
					n_dim = content[1].shape[1]
					data = np.zeros((mini_batch, n_frame, n_dim))
				content_name.append(content[0])
				cnt += 1
				data[k,:,:] = content[1] # load data with (batches, lens, dims)
				label.append(label_dict[content[0]])
				label_len.append(len(label_dict[content[0]]))
			batch_label, batch_mask = label2state(label, label_len, unit2state_dict, blank_symbol)
			yield np.array(data).astype('float32'), batch_label, batch_mask, content_name
		
		if batch_size < batch:
			cnt = 0
			content_name = []
			label = []
			label_len = []
			left_index = xrange(batch_size, batch)

			for k in xrange(len(left_index)):
				content = streams.next()
				if cnt == 0:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
					n_frame = content[1].shape[0]
					n_dim = content[1].shape[1]
					data = np.zeros((len(left_index), n_frame, n_dim))
				content_name.append(content[0])
				cnt += 1
				data[k,:,:] = content[1] # load data with (batches, lens, dims)
				label.append(label_dict[content[0]])
				label_len.append(len(label_dict[content[0]]))
			batch_label, batch_mask = label2state(label, label_len, unit2state_dict, blank_symbol)
			yield np.array(data).astype('float32'), batch_label, batch_mask, content_name				

def shuffle_batch_state_phoneme_data(pkl_data, open_method, sample_index, 
								mini_batch, label_dict, blank_symbol, unit2state_dict):
	streams = stream_file(pkl_data,open_method)
	batch_data = []
	for key in sample_index.keys():

		batch = len(sample_index[key])

		batch_size = 0
		while batch_size + mini_batch <= batch:
			content_name = []
			cnt = 0
			label = []
			label_len = []
			start_n = batch_size
			end_n = batch_size + mini_batch

			batch_size = end_n
			for k in xrange(mini_batch):
				content = streams.next()
				if cnt == 0:
					n_frame = content[1].shape[0]
					n_dim = content[1].shape[1]
					data = np.zeros((mini_batch, n_frame, n_dim))
				content_name.append(content[0])
				cnt += 1
				data[k,:,:] = content[1] # load data with (batches, lens, dims)
				label.append(label_dict[content[0]])
				label_len.append(len(label_dict[content[0]]))
			batch_label, batch_mask = label2state(label, label_len, unit2state_dict, blank_symbol)
			batch_data.append([np.array(data).astype('float32'), batch_label,batch_mask,content_name])
		
		if batch_size < batch:
			cnt = 0
			content_name = []
			label = []
			label_len = []
			left_index = xrange(batch_size, batch)

			for k in xrange(len(left_index)):
				content = streams.next()
				if cnt == 0:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
					n_frame = content[1].shape[0]
					n_dim = content[1].shape[1]
					data = np.zeros((len(left_index), n_frame, n_dim))
				content_name.append(content[0])
				cnt += 1
				data[k,:,:] = content[1] # load data with (batches, lens, dims)
				label.append(label_dict[content[0]])
				label_len.append(len(label_dict[content[0]]))
			batch_label, batch_mask = label2state(label, label_len, unit2state_dict, blank_symbol)
			batch_data.append([np.array(data).astype('float32'), batch_label,batch_mask,content_name])
	return batch_data

def iterate_batch_state_normalized_phoneme_data(pkl_data, open_method, sample_index, 
								mini_batch, label_dict, blank_symbol, unit2state_dict,
								train_mean, train_std):
	streams = stream_file(pkl_data,open_method)
	for key in sample_index.keys():

		batch = len(sample_index[key])

		batch_size = 0
		while batch_size + mini_batch <= batch:
			content_name = []
			cnt = 0
			label = []
			label_len = []
			start_n = batch_size
			end_n = batch_size + mini_batch

			batch_size = end_n
			for k in xrange(mini_batch):
				content = streams.next()
				if cnt == 0:
					n_frame = content[1].shape[0]
					n_dim = content[1].shape[1]
					data = np.zeros((mini_batch, n_frame, n_dim))
				content_name.append(content[0])
				cnt += 1
				data[k,:,:] = content[1] # load data with (batches, lens, dims)
				label.append(label_dict[content[0]])
				label_len.append(len(label_dict[content[0]]))
			batch_label, batch_mask = label2state(label, label_len, unit2state_dict, blank_symbol)
			data = np.array(data).astype('float32')
			data = (data - train_mean) / train_std
			yield data, batch_label, batch_mask, content_name
		
		if batch_size < batch:
			cnt = 0
			content_name = []
			label = []
			label_len = []
			left_index = xrange(batch_size, batch)

			for k in xrange(len(left_index)):
				content = streams.next()
				if cnt == 0:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
					n_frame = content[1].shape[0]
					n_dim = content[1].shape[1]
					data = np.zeros((len(left_index), n_frame, n_dim))
				content_name.append(content[0])
				cnt += 1
				data[k,:,:] = content[1] # load data with (batches, lens, dims)
				label.append(label_dict[content[0]])
				label_len.append(len(label_dict[content[0]]))
			batch_label, batch_mask = label2state(label, label_len, unit2state_dict, blank_symbol)
			data = np.array(data).astype('float32')
			data = (data - train_mean) / train_std
			yield data, batch_label, batch_mask, content_name				

def iterate_batch_phoneme_data(pkl_data, open_method, sample_index, 
								mini_batch, label_dict, blank_symbol):
	streams = stream_file(pkl_data,open_method)
	for key in sample_index.keys():

		batch = len(sample_index[key])

		batch_size = 0
		while batch_size + mini_batch <= batch:
			content_name = []
			cnt = 0
			label = []
			label_len = []
			start_n = batch_size
			end_n = batch_size + mini_batch

			batch_size = end_n
			for k in xrange(mini_batch):
				content = streams.next()
				if cnt == 0:
					n_frame = content[1].shape[0]
					n_dim = content[1].shape[1]
					data = np.zeros((mini_batch, n_frame, n_dim))
				content_name.append(content[0])
				cnt += 1
				data[k,:,:] = content[1] # load data with (batches, lens, dims)
				label.append(label_dict[content[0]])
				label_len.append(len(label_dict[content[0]]))
			batch_label, batch_mask = label2mask(label, label_len, blank_symbol)
			yield np.array(data).astype('float32'), batch_label, batch_mask, content_name
		
		if batch_size < batch:
			cnt = 0
			content_name = []
			label = []
			label_len = []
			left_index = xrange(batch_size, batch)

			for k in xrange(len(left_index)):
				content = streams.next()
				if cnt == 0:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
					n_frame = content[1].shape[0]
					n_dim = content[1].shape[1]
					data = np.zeros((len(left_index), n_frame, n_dim))
				content_name.append(content[0])
				cnt += 1
				data[k,:,:] = content[1] # load data with (batches, lens, dims)
				label.append(label_dict[content[0]])
				label_len.append(len(label_dict[content[0]]))
			batch_label, batch_mask = label2mask(label, label_len, blank_symbol)

			yield np.array(data).astype('float32'), batch_label, batch_mask, content_name				

def data_loader_no_label(train_scp, feats_scp, scp_len,
				spk2utt):

	spk_seg_label, num_spks = spklabel(spk2utt)
	sample_index = lenscp2trainIndex(feats_scp, scp_len, train_scp)

	return sample_index, spk_seg_label, num_spks

def iterate_batch_data_no_label(pkl_data, open_method, sample_index, 
								spk_seg_label, num_spks, mini_batch):
	streams = stream_file(pkl_data,open_method)
	for key in sample_index.keys():

		batch = len(sample_index[key])

		batch_size = 0
		while batch_size + mini_batch <= batch:
			content_name = []
			seg_length = []
			cnt = 0
			spk_sub_label = []
			start_n = batch_size
			end_n = batch_size + mini_batch

			batch_size = end_n
			for k in xrange(mini_batch):
				content = streams.next()
				if cnt == 0:
					n_frame = content[1].shape[0]
					n_dim = content[1].shape[1]
					data = np.zeros((mini_batch, n_frame, n_dim))
				content_name.append(content[0])
				cnt += 1
				seg_length.append(content[1].shape[0]) 
				data[k,:,:] = content[1] # load data with (batches, lens, dims)
				spk_sub_label.append(spk_seg_label[content[0]])
			spk_sub_label = np.array(spk_sub_label).astype('int32')
			classes, labels = np.unique(spk_sub_label, return_inverse=True)
			one_hot_label = one_hot(spk_sub_label, num_spks)
			yield np.array(data).astype('float32'), np.array(one_hot_label).astype('float32'), np.array(seg_length).astype('int32'), content_name			
		
		if batch_size < batch:
			seg_length = []
			cnt = 0
			spk_sub_label = []
			content_name = []

			left_index = xrange(batch_size, batch)

			for k in xrange(len(left_index)):
				content = streams.next()
				if cnt == 0:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
					n_frame = content[1].shape[0]
					n_dim = content[1].shape[1]
					data = np.zeros((len(left_index), n_frame, n_dim))
				content_name.append(content[0])
				cnt += 1
				data[k,:,:] = content[1] # load data with (batches, lens, dims)
				spk_sub_label.append(spk_seg_label[content[0]])
				seg_length.append(content[1].shape[0])

			spk_sub_label = np.array(spk_sub_label).astype('int32')
			classes, labels = np.unique(spk_sub_label, return_inverse=True)
			one_hot_label = one_hot(spk_sub_label, num_spks)

			yield np.array(data).astype('float32'), np.array(one_hot_label).astype('float32'), np.array(seg_length).astype('int32'), content_name
				
def stream_read(filename, open_method):
	spk_data = OrderedDict()
	spk_name = OrderedDict()
	with open_method(filename,'rb') as fd:
		try:
			cnt = 0
			while True:
				content = pkl.load(fd)
				content_name = content[0]
				name = content_name.split('_')[0] + '_sre16'
				data = np.array(content[1]).astype('float32') # load data with (lens, dims)
				spk_data[content_name] = data
				if spk_name.has_key(name):
					spk_name[name] = np.concatenate((spk_name[name],data),axis=0)
				else:
					spk_name[name] = data
				cnt += 1		
		except EOFError: pass
		return spk_data, spk_name

def merge_features(pkl_data, open_method, min_lens, frame_lens):
	
	spk_data, spk_name = stream_read(pkl_data, open_method)
	print 'begin to merge features'
	spk_seg = OrderedDict()
	spk2utt = OrderedDict()
	for spk in spk_name.keys():
		data = spk_name[spk]
		spk2utt[spk] = []
		#print data.shape
		cnt = 0
		accumulate_lens = 0
		if data.shape[0] <= min_lens:
			begt_frame = 0
			endt_frame = data.shape[0]
			begt_time = begt_frame * frame_lens
			endt_time = endt_frame * frame_lens
			begt = time2string(begt_time)
			endt = time2string(endt_time)
			spk_seg_name = spk + '_' + begt + '-' + endt
			spk2utt[spk].append(spk_seg_name)
			spk_seg[spk_seg_name] = data
		elif data.shape[0] > min_lens:
			while accumulate_lens + min_lens < data.shape[0]:
				if data.shape[0] - (accumulate_lens + min_lens) > min_lens:
					begt_frame = accumulate_lens
					endt_frame = accumulate_lens + min_lens - 1
				elif data.shape[0] - (accumulate_lens + min_lens) <= min_lens:
					begt_frame = accumulate_lens
					endt_frame = data.shape[0]
				begt_time = begt_frame * frame_lens
				endt_time = endt_frame * frame_lens
				begt = time2string(begt_time)
				endt = time2string(endt_time)
				spk_seg_name = spk + '_' + begt + '-' + endt
				spk_seg[spk_seg_name] = data[begt_frame:endt_frame,:] 
				spk2utt[spk].append(spk_seg_name)
				accumulate_lens += min_lens
	return spk_seg, spk2utt

def info_merge(spk_seg, spk2utt, merge_train_scp, 
				merge_len, merge_feats_scp, merge_spk2utt, output_file, remove_dup_spk2utt):
	try:
		fwobj_train_scp = open(merge_train_scp,'w')
		fwobj_len = open(merge_len,'w')
		fwobj_feats_scp = open(merge_feats_scp,'w')
		fwobj_spk2utt = open(merge_spk2utt,'w')
		fwobj_remove_spk = open(remove_dup_spk2utt,'w')
	except IOError:
		print "failed to open to write ", merge_train_scp
		print "failed to open to write ", merge_len
		print "failed to open to write ", merge_feats_scp
		print "failed to open to write ", merge_spk2utt
	else:
		sorted_scp = sorted(spk_seg.items(), key=lambda item:item[1].shape[0])
		for item in sorted_scp:
			fwobj_len.write(str(item[1].shape[0]))
			fwobj_len.write('\n')
		fwobj_len.close()

		for key in spk2utt:
			fwobj_spk2utt.write(key)
			fwobj_spk2utt.write(' ')
			fwobj_remove_spk.write(key)
			fwobj_remove_spk.write(' ')
			fwobj_remove_spk.write(key)
			fwobj_remove_spk.write('\n')
			for item in spk2utt[key]:
				fwobj_spk2utt.write(item)
				fwobj_spk2utt.write(' ')
			fwobj_spk2utt.write('\n')
		fwobj_spk2utt.close()
		fwobj_remove_spk.close()

		for items in sorted_scp:
			item = items[0]
			fwobj_feats_scp.write(item)
			fwobj_feats_scp.write(' ')
			fwobj_feats_scp.write(item)
			fwobj_feats_scp.write('\n')
			fwobj_train_scp.write(item)
			fwobj_train_scp.write(' ')
			fwobj_train_scp.write(item)
			fwobj_train_scp.write('\n')
		fwobj_train_scp.close()
		fwobj_feats_scp.close()
		with gzip.open(output_file,'wb') as f:
			count = 0
			for info in sorted_scp:
				name = info[0]
				feature = info[1]
				pkl.dump((name,feature),f,protocol=2)
				count += 1
				if count % 100 == 0:
					print "Wrote %d utterances to %s"%(count,output_file)
			print "Wrote %d utterances to %s"%(count,output_file)

def iterate_batch_data(pkl_data, open_method, sample_index, 
						label_dict, label_mask, spk_seg_label, num_spks, mini_batch):
	streams = stream_file(pkl_data,open_method)
	for key in sample_index.keys():

		batch = len(sample_index[key])

		batch_size = 0
		while batch_size + mini_batch <= batch:
			label_length = []
			content_name = []
			seg_length = []
			cnt = 0
			spk_sub_label = []
			seg_effect_label = []
			seg_label = []
			start_n = batch_size
			end_n = batch_size + mini_batch

			batch_size = end_n
			for k in xrange(mini_batch):
				content = streams.next()
				if cnt == 0:
					n_frame = content[1].shape[0]
					n_dim = content[1].shape[1]
					data = np.zeros((mini_batch, n_frame, n_dim))
				content_name.append(content[0])
				cnt += 1
				seg_effect_label.append(label_mask[content[0]])
				data[k,:,:] = content[1] # load data with (batches, lens, dims)
				spk_sub_label.append(spk_seg_label[content[0]])
				seg_label.extend(label_dict[content[0]])
				label_length.append(len(label_dict[content[0]]))
				seg_length.append(content[1].shape[0])
			spk_sub_label = np.array(spk_sub_label).astype('int32')
			classes, labels = np.unique(spk_sub_label, return_inverse=True)
			one_hot_label = one_hot(spk_sub_label, num_spks)
			yield np.array(data).astype('float32'), np.array(seg_label).astype('int32'), np.array(label_length).astype('int32'), np.array(seg_length).astype('int32'), np.array(one_hot_label).astype('float32'), np.array(seg_effect_label).astype('float32'), content_name			
		
		if batch_size < batch:
			label_length = []
			seg_length = []
			cnt = 0
			spk_sub_label = []
			seg_effect_label = []
			seg_label = []
			content_name = []

			left_index = xrange(batch_size, batch)

			for k in xrange(len(left_index)):
				content = streams.next()
				if cnt == 0:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
					n_frame = content[1].shape[0]
					n_dim = content[1].shape[1]
					data = np.zeros((len(left_index), n_frame, n_dim))
				content_name.append(content[0])
				cnt += 1
				seg_effect_label.append(label_mask[content[0]])
				data[k,:,:] = content[1] # load data with (batches, lens, dims)
				spk_sub_label.append(spk_seg_label[content[0]])
				seg_label.extend(label_dict[content[0]])
				label_length.append(len(label_dict[content[0]]))
				seg_length.append(content[1].shape[0])

			spk_sub_label = np.array(spk_sub_label).astype('int32')
			classes, labels = np.unique(spk_sub_label, return_inverse=True)
			one_hot_label = one_hot(spk_sub_label, num_spks)

			yield np.array(data).astype('float32'), np.array(seg_label), np.array(label_length).astype('int32'), np.array(seg_length).astype('int32'), one_hot_label, np.array(seg_effect_label).astype('int32'), content_name
				
def iterate_batch_spk_data(pkl_data, feats_scp, scp_len, 
							train_scp, open_method, mini_batch):
	streams = stream_file(pkl_data,open_method)
	sample_index = lenscp2trainIndex(feats_scp, scp_len, train_scp)

	for key in sample_index.keys():

		batch = len(sample_index[key])
		#print batch

		batch_size = 0
		while batch_size + mini_batch <= batch:
			label_length = []
			content_name = []
			seg_length = []
			cnt = 0
			start_n = batch_size
			end_n = batch_size + mini_batch

			batch_size = end_n
			for k in xrange(mini_batch):
				content = streams.next()
				if cnt == 0:
					n_frame = content[1].shape[0]
					n_dim = content[1].shape[1]
					data = np.zeros((mini_batch, n_frame, n_dim))
					#print n_dim, content[0]
				content_name.append(content[0])
				cnt += 1
				data[k,:,:] = content[1] # load data with (batches, lens, dims)
			
			yield content_name, np.array(data).astype('float32')
		
		if batch_size < batch:
			cnt = 0
			content_name = []

			left_index = xrange(batch_size, batch)

			for k in xrange(len(left_index)):
				content = streams.next()
				if cnt == 0:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
					n_frame = content[1].shape[0]
					n_dim = content[1].shape[1]
					data = np.zeros((len(left_index), n_frame, n_dim))
					#print n_dim, content[0]
				content_name.append(content[0])
				cnt += 1
				data[k,:,:] = content[1] # load data with (batches, lens, dims)

			yield content_name, np.array(data).astype('float32')	

	# try:
	# 	while True:
	# 		content = streams.next()
	# 		n_frame = content[1].shape[0]
	# 		n_dim = content[1].shape[1]
			
	# 		content_name = content[0]
				
	# 		data = content[1] # load data with (batches, lens, dims)

	# 		data = data.reshape([1,data.shape[0],data.shape[1]])

	# 		yield content_name, np.array(data).astype('float32')
	# except EOFError: pass

def parse_args():
	"""parse input arguments"""
	parser = argparse.ArgumentParser(description='ctc data testing')
	parser.add_argument('--clean_train_scp',dest='clean_train_scp',help='datapath for ...\
		training vae',default='/mnt/workspace/xuht/TIMIT/dev/train.scp')
	parser.add_argument('--noisy_train_scp',dest='noisy_train_scp',help='datapath for...\
		testing vae',default='/mnt/workspace/xuht/kaldi-trunk/egs/swbd/s5c/noisydata/data/train.scp')
	parser.add_argument('--clean_feats_scp',dest='clean_feats_scp',help='datapath for ...\
		training vae',default='/mnt/workspace/xuht/TIMIT/dev/feats.scp')
	parser.add_argument('--noisy_feats_scp',dest='noisy_feats_scp',help='datapath for...\
		testing vae',default='/mnt/workspace/xuht/kaldi-trunk/egs/swbd/s5c/noisydata/feats.scp')
	parser.add_argument('--scp_clean_len',dest='scp_clean_len',help='datapath for ...\
		training vae',default='/mnt/workspace/xuht/TIMIT/dev/len.tmp')
	parser.add_argument('--scp_noisy_len',dest='scp_noisy_len',help='datapath for...\
		testing vae',default='/mnt/workspace/xuht/kaldi-trunk/egs/swbd/s5c/noisydata/len.tmp')
	parser.add_argument('--clean_spk2utt',dest='clean_spk2utt',help='datapath for ...\
		training vae',default='/mnt/workspace/xuht/TIMIT/dev/spk2utt')
	parser.add_argument('--noisy_spk2utt',dest='noisy_spk2utt',help='datapath for...\
		testing vae',default='/mnt/workspace/xuht/kaldi-trunk/egs/swbd/s5c/noisydata/spk2utt')
	parser.add_argument('--clean_pkl',dest='clean_pkl',help='datapath for',
		default='/mnt/workspace/xuht/TIMIT/train/fbank_cmvn_delta.gzip')
	parser.add_argument('--noisy_pkl',dest='noisy_pkl',help='arklist for reading i-vector',
		default='/mnt/workspace/xuht/kaldi-trunk/egs/swbd/s5c/noisydata/fbank__delta.gzip')
	parser.add_argument('--clean_label',dest='clean_label',help='trials for testing speaker verification',
		default='/mnt/workspace/xuht/TIMIT/dev/text_state')
	parser.add_argument('--noisy_label',dest='noisy_label',help='speakers to utterances',
		default='/mnt/workspace/xuht/kaldi-trunk/egs/swbd/s5c/noisydata/labels.cv')
	parser.add_argument('--merge_train_scp',dest='merge_train_scp',help='speakers to utterances',
		default='/mnt/workspace/xuht/TIMIT/dev/merge_train.scp')
	parser.add_argument('--merge_len',dest='merge_len',help='sre10_original_vad_test_mfcc to utterances',
		default='/mnt/workspace/xuht/TIMIT/dev/merge_len.tmp')
	parser.add_argument('--merge_feats_scp',dest='merge_feats_scp',help='speakers to utterances',
		default='/mnt/workspace/xuht/TIMIT/dev/merge_feats.scp')
	parser.add_argument('--merge_spk2utt',dest='merge_spk2utt',help='speakers to utterances',
		default='/mnt/workspace/xuht/TIMIT/dev/merge_spk2utt')
	parser.add_argument('--merge_outputfile',dest='merge_outputfile',help='speakers to utterances',                       
		default='/mnt/workspace/xuht/TIMIT/dev/merge_outputfile.zip')
	parser.add_argument('--merge_remove_spk2utt',dest='merge_remove_spk2utt',help='speakers to utterances',                       
		default='/mnt/workspace/xuht/TIMIT/dev/merge_remove_spk2utt.scp')
	
	args = parser.parse_args()
	return args

if __name__ == '__main__':
	args = parse_args()
	print args
	clean_train_scp = args.clean_train_scp
	noisy_train_scp = args.noisy_train_scp
	clean_feats_scp = args.clean_feats_scp
	noisy_feats_scp = args.noisy_feats_scp
	scp_clean_len = args.scp_clean_len
	scp_noisy_len = args.scp_noisy_len
	clean_spk2utt = args.clean_spk2utt
	noisy_spk2utt = args.noisy_spk2utt
	clean_pkl = args.clean_pkl
	noisy_pkl = args.noisy_pkl
	clean_label = args.clean_label
	noisy_label = args.noisy_label

	merge_train_scp = args.merge_train_scp
	merge_len = args.merge_len
	merge_feats_scp = args.merge_feats_scp
	merge_spk2utt = args.merge_spk2utt
	merge_outputfile = args.merge_outputfile
	merge_remove_spk2utt = args.merge_remove_spk2utt

	data_load = 'test_cmvn'
	if data_load == 'ctc':
		[sample_index, label_dict, 
		label_mask, spk_seg_label, 
		ctc_label_dims, num_spks] = data_loader(clean_train_scp, clean_feats_scp, scp_clean_len,
											clean_spk2utt, clean_label)

		data_iterative = iterate_batch_data(clean_pkl, gzip.open, sample_index, label_dict, 
										label_mask, spk_seg_label, num_spks, 20)

		for k in xrange(20):
			data, label, label_len, data_len, one_hot_label, seg_efficiency_label, seg_name = data_iterative.next()
		
			print k, data.dtype, label.shape, label_len.shape, seg_efficiency_label, np.argmax(one_hot_label,axis=1)
	 	
	elif data_load == 'spk':

		k = 0
		for batch in iterate_batch_spk_data(clean_pkl, clean_feats_scp, scp_clean_len,
											clean_train_scp, gzip.open, 1):
			seg_name, data= batch
			

	elif data_load == 'merge':
		spk_seg, spkutt = merge_features(clean_pkl, gzip.open, 500, 10)
	 	info_merge(spk_seg, spkutt, merge_train_scp, 
				merge_len, merge_feats_scp, merge_spk2utt, merge_outputfile, merge_remove_spk2utt)

	elif data_load == 'timit':
		[sample_index, label_dict, 
		label_mask, spk_seg_label, 
		ctc_label_dims, num_spks] = data_loader(clean_train_scp, clean_feats_scp, scp_clean_len,
											clean_spk2utt, clean_label)

		cnt = 0
		data_itera = iterate_batch_phoneme_data(clean_pkl, gzip.open, sample_index, 
								1, label_dict, 0)
		data_matrix = []
		data_label = []
		for batch in data_itera:
			data, batch_label, batch_mask, content_name = batch
			cnt += data.shape[0]
			data_matrix.append(data)
			data_label.append(batch_label)
			if content_name[0] == 'mbom0_sx384':
				print content_name
				print data.shape
				print batch_label
				print batch_label.shape
				print data
				break

		sio.savemat('/home/xuht/timit_.mat', {'fbank':np.asarray(data_matrix),'label':np.asarray(data_label)})


	elif data_load == 'test_cmvn':
		t=[]
		label = []
		streams = stream_file(clean_pkl, gzip.open)
		for k in xrange(8):
			data = streams.next()
			print data[0], data[1].shape
			t.append(data[1])
			label.append(data[0])
		sio.savemat('/home/xuht/timit_train_fbank1.mat',{'fbank':np.asarray(t),'label':np.asarray(label)})

	elif data_load == 'test_mvn':
		data, mean, std = extract_mvn(clean_pkl, gzip.open, '/home/xuht/train_info.pkl')
		print data.shape
		print mean, mean.shape
		print std, std.shape
		t = (data-mean)/std
		print t.shape
		print np.mean(t,axis=0)
		print np.std(t,axis=0)

	elif data_load == 'state_test':
		units2state = '/mnt/workspace/xuht/TIMIT/lang_phn/units2state.txt'
		[sample_index, label_dict,
		label_mask, spk_seg_label, 
		ctc_label_dims, spk_nums, 
		unit2state_dict] = data_state_loader(clean_train_scp, clean_feats_scp, scp_clean_len,
												clean_spk2utt, clean_label, units2state)

		all_data = iterate_batch_state_phoneme_data(
								clean_pkl, gzip.open, sample_index, 
								5, label_dict, 'eps', 
								unit2state_dict)
		cnt = 0
		for data  in all_data:
			batch_data, batch_label, batch_mask, content_name = data
			if batch_data.shape[0] == 5:
				print batch_data.shape
				cnt += 1
				print batch_mask
				print batch_label
				break




