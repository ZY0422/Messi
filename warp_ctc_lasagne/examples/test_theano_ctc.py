from gpu_ctc import ctc_gpu, ctc_gpu_costs
import numpy as np
import theano
import theano.tensor as T
import time

label_in = np.array([5,4,3,2,5,4,3,2,1,3,2,1,2,1]).astype('int32')
label_len = np.array([4,5,3,2]).astype('int32')

np.random.seed(0)

yy = T.ivector()
yy_mask = T.matrix()
y_len = T.ivector()
x_len = T.ivector()

x = T.tensor3()

data = np.random.randn(200,4,5).astype('float32')
data_len = np.array([200,200,200,200]).astype('int32')

transform = (np.random.randn(5,6).astype('float32'))
lin_output = T.dot(x, transform)

costs, grads = ctc_gpu(lin_output, x_len, yy, y_len)

theano_grad = T.grad(T.mean(costs), lin_output)

g = theano.function([x, x_len, yy, y_len], theano_grad, allow_input_downcast=True)
ctc_g = theano.function([x, x_len, yy, y_len], grads, allow_input_downcast=True)

gg = g(data, data_len, label_in, label_len)
ctc_gg  = ctc_g(data, data_len, label_in, label_len)
print 'theano grad'
print gg
print 'warp ctc grad'
print ctc_gg