#section support_code_struct

int APPLY_SPECIFIC(gpu_ctc_grad)(CudaNdarray* probs,  
							PyArrayObject* prob_lengths,
							PyArrayObject* labels,
							PyArrayObject* label_lengths,
							CudaNdarray** grads)
{
	int minibatch_size = CudaNdarray_HOST_DIMS(probs)[1]; // since first dimension is lengths * batch_size * dims
	int alphabet_size = CudaNdarray_HOST_DIMS(probs)[2];

	int* probs_sizes_ptr = static_cast<int*> (PyArray_DATA(prob_lengths));
    int* labels_ptr = static_cast<int*> (PyArray_DATA(labels));
    int* label_sizes_ptr = static_cast<int*> (PyArray_DATA(label_lengths));

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	ctcComputeInfo info;
	info.loc = CTC_GPU;
	info.stream = stream;

	int ndim_probs = CudaNdarray_NDIM(probs);
	int* cost_dims = &minibatch_size;

	int* probs_dims = new int[ndim_probs];

	for(size_t i=0; i< ndim_probs; i++)
	{
		probs_dims[i] = CudaNdarray_HOST_DIMS(probs)[i];
	}

	CudaNdarray* my_grads = (CudaNdarray*)CudaNdarray_ZEROS(ndim_probs, probs_dims);
	CudaNdarray* costs = (CudaNdarray*)CudaNdarray_ZEROS(1, cost_dims);

	if( CudaNdarray_HOST_DIMS(costs)[0] != minibatch_size )
	{
		PyErr_SetString(PyExc_RuntimeError, "costs");
		return 1;
	}

	if (CudaNdarray_prep_output(grads, ndim_probs, CudaNdarray_HOST_DIMS(my_grads)) != 0)
	{
		PyErr_SetString(PyExc_RuntimeError, "CudaNdarray_prep_output");
		return 1;
	}	
	
	size_t gpu_alloc_bytes;
	ctcStatus_t workspace_error = get_workspace_size(label_sizes_ptr, probs_sizes_ptr,
													alphabet_size, minibatch_size, 
													info, &gpu_alloc_bytes);
	std::cout << "succeeded in workspace" << std::endl;
	if (CTC_STATUS_SUCCESS != workspace_error)
	{
		PyErr_SetString(PyExc_RuntimeError, "error getting GPU memory space for ctc");
        return 1;
	}

	char *ctc_gpu_workspace;
	cudaMalloc(&ctc_gpu_workspace, gpu_alloc_bytes);

    ctcStatus_t compute_ctc_error = compute_ctc_loss(
    										static_cast<float*>(CudaNdarray_DEV_DATA(probs)), 
    										static_cast<float*>(CudaNdarray_DEV_DATA(my_grads)),
													labels_ptr, 
													label_sizes_ptr,
													probs_sizes_ptr, 
													alphabet_size,
													minibatch_size, 
											static_cast<float*>	(CudaNdarray_DEV_DATA(costs)),
													ctc_gpu_workspace, 
													info);

    if (CudaNdarray_CopyFromCudaNdarray(*grads, my_grads))
	{
		PyErr_SetString(PyExc_RuntimeError, "CudaNdarray_CopyFromCudaNdarray");
		return 1;
	}

    if (CTC_STATUS_SUCCESS != compute_ctc_error)
    {
    	PyErr_SetString(PyExc_RuntimeError, "error computing ctc loss");
        return 1;
    }

    Py_DECREF(probs_sizes_ptr);
    Py_DECREF(labels_ptr);
    Py_DECREF(label_sizes_ptr);
    Py_DECREF(labels_ptr);
    Py_DECREF(label_sizes_ptr);
    Py_DECREF(probs_dims);
   	Py_DECREF(cost_dims);
    Py_DECREF(costs);
    Py_DECREF(my_grads);

    cudaFree(ctc_gpu_workspace);
    cudaStreamDestroy(stream);

    return 0;
}



                   



