Running Networks on the GPU
"""""""""""""""""""""""""""

The previous documentation details how to construct and simulate networks which run on a conventional CPU. In this
document, our focus will turn to simulation on GPUs.

Hardware Requirements
======================

SNS-Toolbox is built on top of :code:`torch`, so GPU simulation is restricted to CUDA-compatible graphics cards.

Using Torch
===========

Building a network for execution is very similar to the process presented in the rest of the documentation/tutorials.
However instead of using :code:`list` or :code:`np.ndarray` objects when designing a network, :code:`torch.Tensor`
objects should always be used instead. Syntax for using tensors is nearly the same as using numpy arrays.
::
    import torch
    import numpy as np

    # Make a basic tensor
    a = torch.tensor([1,2,3,4,5])

    # Make a 5x3 element tensor of zeros
    b = torch.zeros([5,3])

    # Make a 3x5 tensor of ones
    c = torch.ones([3,5])

    # Convert a numpy array to a torch tensor
    old = np.array([5,4,3,2,1])
    new = torch.from_numpy(old)

For the full list of operations available with torch tensors, please consult the
`PyTorch documentation <https://pytorch.org/docs/stable/torch.html>`_.

Compiling a Network
===================

In order to compile a network such that it runs on the GPU, the :code:`device` flag must be set to :code:`'cuda'`.
::
    model = net.compile(backend='torch`, device='cuda')
    model_sparse = net.compile(backend='sparse', device='cuda')

Note that GPU support is only available using the :code:`torch` or :code:`sparse` backends. If simulating on a machine
with multiple GPU cards, set the device to :code:`cuda:i` where :code:`i` is the index of the GPU, starting from 0.

Simulating a Network
====================

Below is sample code for simulating a model on the GPU. Note that the network is stored in GPU memory, so any variables
stored on the CPU must be transferred to/from the GPU to interact with the model.
::
    # Set simulation parameters
    dt = 0.01
    t_max = 50

    # Initialize a vector of timesteps
    t = np.arange(0, t_max, dt)

    inputs = torch.zeros([len(t),net.get_num_inputs_actual()],device='cuda')+20.0  # Input vector must be 2d, even if second dimension is 1
    data = torch.zeros([len(t),net.get_num_outputs_actual()], device='cuda')

    for i in range(len(t)):
        data[i, :] = model(inputsTorch[i, :])

    data = torch.transpose(dataTorch,0,1)
    data = data.to('cpu')   # Move the data from the GPU to the CPU so it can be plotted