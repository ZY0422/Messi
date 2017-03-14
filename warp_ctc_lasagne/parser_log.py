import numpy as np
from itertools import izip
import random
import argparse

from itertools import izip
from collections import OrderedDict

import matplotlib.pyplot as plt
import scipy.io as sio

def parser_log(log_path):
	try:
		# epoch 0 dev ctc-cost 960.776842728 dev-per 1.29443411265
		# relative improvement 0.469103629393 total utterance 3696
		# epoch 0 time 1298.11581802 ctc-cost 146.910362939
		frobj_log = open(log_path,'r')
	except IOError:
		print "failed to read ", log_path
	else:
		lines = frobj_log.readlines()
		train_info = []
		dev_info = []
		for eachline in lines:
			content = eachline.strip().split()
			if content[0] == 'epoch' and len(content) >= 4:
				if len(content) == 6:
					# train information
					train_info.append(float(content[-1]))
				elif len(content) == 7:
					dev_info.append([float(content[4]),float(content[-1])])

		train_array = np.array(train_info)
		dev_array = np.array(dev_info)

		return train_array, dev_array

path = '/mnt/workspace/xuht/markvo-ctc/markov-ctc-summarize/two_state/'
log_path = '_logging_2017-02-25-094756.log'
train, dev = parser_log(path+log_path)
sio.savemat(path+'train_dev_094756.mat',{'train':train,'dev':dev})
plt.plot(np.arange(dev.shape[0]),dev[:,1])
plt.show()

