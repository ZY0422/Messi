#section support_code_struct

int APPLY_SPECIFIC(gpu_ctc)(CudaNdarray* probs,  
							PyArrayObject* prob_lengths,
							PyArrayObject* labels,
							PyArrayObject* label_lengths,
							CudaNdarray** costs,
							CudaNdarray** grads)
{
	int minibatch_size = CudaNdarray_HOST_DIMS(probs)[1]; // since first dimension is lengths * batch_size * dims
	int alphabet_size = CudaNdarray_HOST_DIMS(probs)[2];

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	ctcComputeInfo info;
	info.loc = CTC_GPU;
	info.stream = stream;

	int ndim_probs = CudaNdarray_NDIM(probs);

	int* probs_dims = new int[ndim_probs];

	for(size_t i=0; i< ndim_probs; i++)
	{
		probs_dims[i] = CudaNdarray_HOST_DIMS(probs)[i];
	}

	*grads = (CudaNdarray*)CudaNdarray_ZEROS(ndim_probs, probs_dims);
	*costs = (CudaNdarray*)CudaNdarray_ZEROS(1, &minibatch_size);

	size_t gpu_alloc_bytes;
	ctcStatus_t workspace_error = get_workspace_size(static_cast<int*> (PyArray_DATA(label_lengths)), 
													static_cast<int*> (PyArray_DATA(prob_lengths)),
													alphabet_size, 
													minibatch_size, 
													info, 
													&gpu_alloc_bytes);
	if (CTC_STATUS_SUCCESS != workspace_error)
	{
		PyErr_SetString(PyExc_RuntimeError, "error getting GPU memory space for ctc");
        return 1;
	}

	char *ctc_gpu_workspace;
	cudaMalloc(&ctc_gpu_workspace, gpu_alloc_bytes);

    ctcStatus_t compute_ctc_error = compute_ctc_loss(
    												CudaNdarray_DEV_DATA(probs), 
    												CudaNdarray_DEV_DATA(*grads),
													static_cast<int*> (PyArray_DATA(labels)), 
													static_cast<int*> (PyArray_DATA(label_lengths)),
													static_cast<int*> (PyArray_DATA(prob_lengths)), 
													alphabet_size,
													minibatch_size, 
													CudaNdarray_DEV_DATA(*costs),
													ctc_gpu_workspace, 
													info);

    if (CTC_STATUS_SUCCESS != compute_ctc_error)
    {
    	PyErr_SetString(PyExc_RuntimeError, "error computing ctc loss");
        return 1;
    }

	
    Py_DECREF(probs_dims);

    cudaFree(ctc_gpu_workspace);
    cudaStreamDestroy(stream);

    return 0;
}

                   



