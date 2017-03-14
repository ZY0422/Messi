from gpu_ctc import ctc_gpu, ctc_gpu_costs
import numpy as np
import theano
import theano.tensor as T
import time

y_in = np.array([[4,3,2,1,-1],[4,3,2,1,0],[2,1,0,-1,-1],[1, 0, -1, -1, -1]]).astype('int32')
y_mask_in = np.array([[1,1,1,1,0],[1,1,1,1,1],[1,1,1,0,0],[1,1,0,0,0]]).astype('float32')

label_in = np.array([5,4,3,2,5,4,3,2,1,3,2,1,2,1]).astype('int32')
label_len = np.array([4,5,3,2]).astype('int32')

np.random.seed(0)

yy = T.ivector()
yy_mask = T.matrix()
y_len = T.ivector()
x_len = T.ivector()

x = T.tensor3()
data = np.random.randn(5, 4, 5).astype('float32')
data_len = np.array([5,5,5,5]).astype('int32')

transform = theano.shared(np.random.randn(5, 6).astype(np.float32))
lin_output = T.dot(x, transform)

output_costs = ctc_gpu_costs(lin_output, x_len, yy, y_len)
costs = T.mean(output_costs)
param_grads = T.grad(costs,transform)
input_grad = T.grad(costs, lin_output)

f_cost = theano.function([x,x_len, yy, y_len],[costs, output_costs])
f_grad = theano.function([x,x_len, yy, y_len],param_grads)
f_in_grad = theano.function([x,x_len, yy, y_len],input_grad)

print f_cost(data, data_len, label_in, label_len)
print f_grad(data, data_len, label_in, label_len)
print f_in_grad(data, data_len, label_in, label_len)