from gpu_ctc import ctc_gpu, ctc_gpu_costs
import numpy as np
import theano
import theano.tensor as T
import time

acts = np.array([[[1,2,3,4,5],[-5,-4,-3,-2,-1],[0,0,0,0,0]],
                 [[6,7,8,9,10],[-10,-9,-8,-7,-6],[0,0,0,0,0]],
                [[11,12,13,14,15],[-15,-14,-13,-12,-11],[0,0,0,0,0]]]).astype('float32')

print acts[:,0,:]
print acts.shape
print acts.dtype
print acts
print np.prod(acts.shape)

act = T.tensor3()

labels = np.array([3,3,2,3,1]).astype('int32')
label_lens = np.array([2,2,1]).astype('int32')
act_lens = np.array([3,3,1]).astype('int32')

costs = ctc_gpu_costs(act, act_lens, labels, label_lens)
a=T.grad(T.mean(costs),act)
f=theano.function([act],[costs,a])
start_time = time.time()
for k in xrange(1000):
  print f(acts)
end_time = time.time()
print 'total time is ', end_time - start_time












