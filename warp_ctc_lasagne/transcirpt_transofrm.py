# -*- coding: utf-8 -*-
import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict

def unit2statemap_sil(units, unit2state, state_num, 
					add_eps_units, eps_state_num, eps_num):
	try:
		frobj = open(units,'r')
		fwobj = open(unit2state, 'w')
	except IOError:
		print "failed to open ", units, " for reading"
		print "failed to open ", unit2state, " for writing"
	else:
		info = frobj.readlines()
		uint2state = OrderedDict()
		cnt = 0
		uint2state[add_eps_units] = list(np.arange(eps_state_num) + eps_num)
		cnt += (uint2state[add_eps_units][-1]+eps_state_num)
		for eachline in info:
			content = eachline.strip().split()
			uint2state[content[0]] = list(np.arange(state_num) + cnt)
			cnt += state_num
		for key in uint2state.keys():
			fwobj.write(str(key))
			fwobj.write(' ')
			for index in uint2state[key]:
				fwobj.write(str(index))
				fwobj.write(' ')
			fwobj.write('\n')
		frobj.close()
		fwobj.close()
		return uint2state

def unit2statemap(units, unit2state, state_num, add_eps_units):
	try:
		frobj = open(units,'r')
		fwobj = open(unit2state, 'w')
	except IOError:
		print "failed to open ", units, " for reading"
		print "failed to open ", unit2state, " for writing"
	else:
		info = frobj.readlines()
		uint2state = OrderedDict()
		cnt = 0
		for eachline in info:
			content = eachline.strip().split()
			uint2state[content[0]] = list(np.arange(state_num) + cnt)
			cnt += state_num
		if add_eps_units is not None:
			uint2state[add_eps_units] = list(np.arange(state_num) + cnt)
		for key in uint2state.keys():
			fwobj.write(str(key))
			fwobj.write(' ')
			for index in uint2state[key]:
				fwobj.write(str(index))
				fwobj.write(' ')
			fwobj.write('\n')
		frobj.close()
		fwobj.close()
		return uint2state

def transcript2state(transcript, unit2state, transcript_state):
	try:
		frobj_trans = open(transcript, 'r')
		frobj_maps = open(unit2state, 'r')
		fwobj = open(transcript_state, 'w')
	except IOError:
		print "failed to open ", transcript, " for reading"
		print "failed to open ", unit2state, " for reading"
		print "failed to open ", transcript_state, " for writing"
	else:
		trans_info = frobj_trans.readlines()
		unit2state_info = frobj_maps.readlines()
		uint2state_dict = OrderedDict()
		transcript2state_dict = OrderedDict()
		state_dict = OrderedDict()
		for eachline in unit2state_info:
			content = eachline.strip().split()
			l = []
			for item in content[1:]:
				l.append(int(item))
			uint2state_dict[content[0]] = l
		for eachline in trans_info:
			content = eachline.strip().split()
			transcript2state_dict[content[0]] = content[1:]
		for key in transcript2state_dict.keys():
			info = []
			for item in transcript2state_dict[key]:
				info.extend((uint2state_dict[item]))
			state_dict[key] = info
		for key in state_dict.keys():
			fwobj.write(str(key))
			fwobj.write(' ')
			for item in state_dict[key]:
				fwobj.write(str(item))
				fwobj.write(' ')
			fwobj.write('\n')
		frobj_trans.close()
		frobj_maps.close()
		fwobj.close()

if __name__ == '__main__':
	units = '/mnt/workspace/xuht/TIMIT/lang_phn/units.txt'
	unit2state = '/mnt/workspace/xuht/TIMIT/lang_phn/units2state_2.txt'
	uint2state_dict = unit2statemap(units, unit2state, 2, None)
	print uint2state_dict
	transcript = '/mnt/workspace/xuht/TIMIT/dev/text'
	transcript_state = '/mnt/workspace/xuht/TIMIT/dev/text_state_2'
	

	# uint2state_dict = unit2statemap_sil(units, unit2state, 2, 
	# 				'eps', 1, 0)
	# print uint2state_dict
	transcript2state(transcript, unit2state, transcript_state)
