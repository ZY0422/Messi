#section support_code_struct

int APPLY_SPECIFIC(cpu_ctc)(PyArrayObject* probs, 
							PyArrayObject* prob_lengths,
							PyArrayObject* labels,
							PyArrayObject* label_lengths,
							PyArrayObject** costs,
							PyArrayObject** grads)
{
	
	int minibatch_size = int(PyArray_DIM(probs, 1)); // since first dimension is lengths * batch_size * dims
	int alphabet_size = int(PyArray_DIM(probs,2));
	int* probs_sizes_ptr = static_cast<int*> (PyArray_DATA(prob_lengths));
    int* labels_ptr = static_cast<int*> (PyArray_DATA(labels));
    int* label_sizes_ptr = static_cast<int*> (PyArray_DATA(label_lengths));

	ctcComputeInfo info;
	info.loc = CTC_CPU;
	info.num_threads = 1;

	if (NULL == PyArray_DATA(probs))
	{
		return -1;
	}

	int ndim_probs = PyArray_NDIM(probs);

	npy_intp* cost_dims[] = {minibatch_size};

	*grads = (PyArrayObject*)PyArray_Zeros(ndim_probs, PyArray_DIMS(probs), 
											PyArray_DescrFromType(NPY_FLOAT32), 0);
	*costs = (PyArrayObject*)PyArray_Zeros(1, cost_dims, 
											PyArray_DescrFromType(NPY_FLOAT32), 0);

	size_t cpu_alloc_bytes;
	ctcStatus_t workspace_error = get_workspace_size(label_sizes_ptr, labels_ptr,
													alphabet_size, probs_sizes_ptr, 
													info, &gpu_alloc_bytes);
	if (CTC_STATUS_SUCCESS != workspace_error)
	{
		PyErr_SetString(PyExc_RuntimeError, "error getting CPU memory space for ctc");
        return -1;
	}

	void* ctc_cpu_workspace = malloc(cpu_alloc_bytes);

    ctcStatus_t compute_ctc_error compute_ctc_loss(
    				static_cast<float*>(PyArray_DATA(probs)), 
    				static_cast<float*>(PyArray_DATA(*grads)),
					labels_ptr, label_sizes_ptr,
					probs_sizes_ptr, alphabet_size,
					minibatch_size, 
					static_cast<float*>(PyArray_DATA(*costs)),
					ctc_gpu_workspace, 
					info);
    if (CTC_STATUS_SUCCESS != compute_ctc_error)
    {
    	PyErr_SetString(PyExc_RuntimeError, "error computing ctc loss");
        return -1;
    }

    delete probs_sizes_ptr;
    delete labels_ptr;
    delete label_sizes_ptr;
    delete labels_ptr;
    delete label_sizes_ptr;

    free(ctc_cpu_workspace);

    return 0;
}