import os
import numpy
import warnings

from six import integer_types

import theano
from theano import Apply, tensor, config, Variable
from theano.scalar import as_scalar, constant, Log
from theano.gradient import DisconnectedType, grad_not_implemented, grad_undefined
from theano.gof import Optimizer, local_optimizer, COp
from theano.gof.type import CDataType
from theano.compile import optdb
from theano.compile.ops import shape_i
from theano.tensor.nnet import LogSoftmax, SoftmaxGrad
from theano.tensor.nnet.abstract_conv import get_conv_output_shape
from theano.tensor.signal.pool import (
	Pool, MaxPoolGrad, AveragePoolGrad)
from theano.sandbox.cuda.type import CudaNdarrayType

from theano.sandbox.cuda import GpuOp, dnn_available
from theano.sandbox.cuda import dnn_version as version
from theano.sandbox.cuda.nvcc_compiler import NVCC_compiler

from theano.sandbox.cuda.basic_ops import (as_cuda_ndarray_variable,
											host_from_gpu,
											gpu_contiguous, HostFromGpu,
											gpu_alloc, GpuAlloc,
											gpu_alloc_empty, GpuAllocEmpty,
											GpuElemwise)

# class CTCGPUGRAD(GpuOp,COp):
# 	"""
# 		warp-ctc gpu ops, using gpu_ctc func for gradients
# 	"""
# 	__props__ = ()
# 	check_broadcast = False

# 	def __init__(self):
# 		COp.__init__(self, ["warp_ctc_grad.c"],
#                      "APPLY_SPECIFIC(gpu_ctc_grad)")

# 	def c_headers(self):
# 		return ['ctc.h']

# 	def c_header_dirs(self):
# 		return ['/usr/local/lib/warp-ctc-master/include']

# 	def c_libraries(self):
# 		return ['warpctc']

# 	def c_lib_dirs(self):
# 		return ['/usr/local/lib/warp-ctc-master/build']

# 	def c_compiler(self):
# 		return NVCC_compiler

# 	def do_constant_folding(self, node):
# 		return False

# 	def c_compile_args(self):
# 		return ['-Wl,-rpath,' +  '/usr/local/lib/warp-ctc-master/build']

# 	def make_node(self, acts, acts_lens, labels, label_lens):
# 		acts = as_cuda_ndarray_variable(acts)
# 		acts_lens = theano.tensor.as_tensor_variable(acts_lens)
# 		labels = theano.tensor.as_tensor_variable(labels)
# 		label_lens = theano.tensor.as_tensor_variable(label_lens)

# 		if acts.type.ndim != 3:
# 			raise TypeError('acts must be 3D tensor')
		
# 		return Apply(self, [acts, acts_lens, labels, label_lens],
# 					[acts.type()])
# ctc_gpu_grad = CTCGPUGRAD()

class CTCGPU(GpuOp,COp):
	"""
		warp-ctc gpu ops, using gpu_ctc func for gradients
	"""
	__props__ = ()
	check_broadcast = False

	def __init__(self):
		COp.__init__(self, ["warp_gpu_ctc.c"],
                     "APPLY_SPECIFIC(gpu_ctc)")

	def c_headers(self):
		return ['ctc.h']

	def c_header_dirs(self):
		return ['/usr/local/lib/warp-ctc-master/include']

	def c_libraries(self):
		return ['warpctc']

	def c_lib_dirs(self):
		return ['/usr/local/lib/warp-ctc-master/build']

	def c_compiler(self):
		return NVCC_compiler

	def do_constant_folding(self, node):
		return False

	def c_compile_args(self):
		return ['-Wl,-rpath,' +  '/usr/local/lib/warp-ctc-master/build']

	def make_node(self, acts, acts_lens, labels, label_lens):
		acts = as_cuda_ndarray_variable(acts)
		acts_lens = theano.tensor.as_tensor_variable(acts_lens)
		labels = theano.tensor.as_tensor_variable(labels)
		label_lens = theano.tensor.as_tensor_variable(label_lens)

		if acts.type.ndim != 3:
			raise TypeError('acts must be 3D tensor')
		
		return Apply(self, [acts, acts_lens, labels, label_lens],
					[CudaNdarrayType(broadcastable=acts_lens.broadcastable,
                                      dtype=acts.type.dtype)(),acts.type()])

	def grad(self, inputs, output_grads):
		acts, acts_lens, labels, label_lens = inputs
		
		_,gradients = self(acts, acts_lens, labels, label_lens)

		return [gradients, 
				grad_undefined(self, 1, inputs[1]),
				grad_undefined(self, 2, inputs[2]),
				grad_undefined(self, 3, inputs[3])]

ctc_gpu = CTCGPU()

def ctc_gpu_costs(acts, acts_lens, labels, label_lens):
	acts = gpu_contiguous(acts)
	acts_lens = theano.tensor.as_tensor_variable(acts_lens)
	labels = theano.tensor.as_tensor_variable(labels)
	label_lens = theano.tensor.as_tensor_variable(label_lens)

	costs,_ = CTCGPU()(acts, acts_lens, labels, label_lens)

	return costs

	





