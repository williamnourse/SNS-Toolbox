"""
Compare the simulation speed of the different backends
William Nourse
September 9 2021
The correct term is Babes, sir
"""

import numpy as np
import torch
import time
import pickle

from sns_toolbox.design.neurons import SpikingNeuron
from sns_toolbox.design.connections import SpikingSynapse
from sns_toolbox.design.networks import Network
from sns_toolbox.simulate.backends import SNS_Numpy, SNS_Torch, SNS_Torch_Large

"""
########################################################################################################################
NEURON AND SYNAPSE DEFINITIONS
"""
current = 10.0
globalStart = time.time()
spike = SpikingNeuron(name='m<0', threshold_proportionality_constant=-1, color='aquamarine')
spikeBias = SpikingNeuron(name='bias', threshold_proportionality_constant=-1, color='aquamarine', bias=current)
spikeExcite = SpikingSynapse(name='Excitatory Spiking')
print('Finished type definition. Running for %f sec'%(time.time()-globalStart))

"""
########################################################################################################################
TEST SETUP
"""
numSamples = 100
numNeurons = np.logspace(1,4,num=numSamples)
dt = 0.01
tMax = 10
t = np.arange(0,tMax,dt)
npTimes = np.zeros([numSamples,len(t)])
torchCPUTimes = np.zeros([numSamples,len(t)])
torchGPUTimes = np.zeros([numSamples,len(t)])
torchGPUTransferTimes = np.zeros([numSamples,len(t)])
torchGPUSparseTimes = np.zeros([numSamples,len(t)])

print('Finished test setup. Running for %f sec'%(time.time()-globalStart))

for num in range(numSamples):
    print('%i Neurons. Running for %f sec' % (numNeurons[num],time.time() - globalStart))
    net = Network()
    numIns = int(0.08 * numNeurons[num]) + 1
    numOuts = int(0.12 * numNeurons[num])
    numSyn = int(np.sqrt(numNeurons[num]))
    numRest = int(numNeurons[num]) - numIns - numSyn - numOuts
    net.add_population(spike, numIns, name='ins')  # in puppy, num_inputs = 8% of network
    net.add_population(spikeBias, numOuts, name='outs')  # in puppy, num_outputs = 12% of network
    net.add_population(spikeBias, numSyn, name='connected')  # in puppy, numSyn = num_neurons
    net.add_population(spikeBias, numRest, name='rest')  # rest of the network
    net.add_synapse(spikeExcite, 'connected', 'connected')
    net.add_input('Input')
    net.addInputConnection(1.0, 'Input', 'ins')
    net.add_output('outs')

    # Numpy
    npModel = SNS_Numpy(net,dt=dt)
    npInput = np.array([current])
    for i in range(len(t)):
        print('%i Neurons Numpy Step %i/%i'%(numNeurons[num],i+1,len(t)))
        stepStart = time.time()
        _ = npModel.forward(npInput)
        stepStop = time.time()
        npTimes[num,i] = stepStop-stepStart
    print('Finished Numpy. Running for %f sec' % (time.time() - globalStart))

    # Torch CPU
    torchCPUModel = SNS_Torch(net,dt=dt,device='cpu')
    torchCPUInput = torch.tensor([current],dtype=torch.float64,device='cpu')
    for i in range(len(t)):
        print('%i Neurons Torch CPU Step %i/%i'%(numNeurons[num],i+1,len(t)))
        stepStart = time.time()
        _ = torchCPUModel.forward(torchCPUInput)
        stepStop = time.time()
        torchCPUTimes[num, i] = stepStop - stepStart
    print('Finished Torch CPU. Running for %f sec' % (time.time() - globalStart))

    try:
        # Torch GPU
        # print('Before network created')
        # print('GPU Memory Allocated: %d , Reserved: %d'%(torch.cuda.memory_allocated(),torch.cuda.memory_reserved()))
        torchGPUModel = SNS_Torch(net, dt=dt, device='cuda')
        torchGPUInput = torch.tensor([current],dtype=torch.float64,device='cuda')
        # print('CUDA Model Made')
        # print('GPU Memory Allocated: %d , Reserved: %d' % (torch.cuda.memory_allocated(), torch.cuda.memory_reserved()))
        for i in range(len(t)):
            print('%i Neurons Torch GPU Step %i/%i'%(numNeurons[num],i+1,len(t)))
            stepStart = time.time()
            _ = torchGPUModel.forward(torchGPUInput)
            stepStop = time.time()
            torchGPUTimes[num, i] = stepStop - stepStart
        del torchGPUModel
        del torchGPUInput
        del _
        # print('CUDA Models deleted')
        # print('GPU Memory Allocated: %d , Reserved: %d' % (torch.cuda.memory_allocated(), torch.cuda.memory_reserved()))
        torch.cuda.empty_cache()
        # print('CUDA cache cleared')
        # print('GPU Memory Allocated: %d , Reserved: %d' % (torch.cuda.memory_allocated(), torch.cuda.memory_reserved()))
        print('Finished Torch GPU. Running for %f sec' % (time.time() - globalStart))
    except:
        print('Torch GPU Cuda compilation failed. Running for %f sec' % (time.time() - globalStart))
        for i in range(len(t)):
            torchGPUTimes[num, i] = 0

    try:
        # Torch GPU with memory transfer at each step
        # print('Before network created')
        # print('GPU Memory Allocated: %d , Reserved: %d'%(torch.cuda.memory_allocated(),torch.cuda.memory_reserved()))
        torchGPUTransferModel = SNS_Torch(net, dt=dt, device='cuda')
        torchGPUTransferInput = torch.tensor([current], dtype=torch.float64, device='cpu')
        # print('CUDA Model Made')
        # print('GPU Memory Allocated: %d , Reserved: %d' % (torch.cuda.memory_allocated(), torch.cuda.memory_reserved()))
        for i in range(len(t)):
            print('%i Neurons Torch GPU Transfer Step %i/%i'%(numNeurons[num],i+1,len(t)))
            stepStart = time.time()
            a = torchGPUTransferModel.forward(torchGPUTransferInput.cuda())
            _ = a.cpu()
            stepStop = time.time()
            torchGPUTransferTimes[num, i] = stepStop - stepStart
        del torchGPUTransferModel
        del torchGPUTransferInput
        del a
        # print('CUDA Models deleted')
        # print('GPU Memory Allocated: %d , Reserved: %d' % (torch.cuda.memory_allocated(), torch.cuda.memory_reserved()))
        torch.cuda.empty_cache()
        # print('CUDA cache cleared')
        # print('GPU Memory Allocated: %d , Reserved: %d' % (torch.cuda.memory_allocated(), torch.cuda.memory_reserved()))
        print('Finished Torch GPU with memory transfer. Running for %f sec' % (time.time() - globalStart))
    except:
        print('GPU Memory Transfer Cuda compilation failed. Running for %f sec' % (time.time() - globalStart))
        for i in range(len(t)):
            torchGPUTransferTimes[num, i] = 0

    try:
        # Torch Sparse GPU
        # print('Before network created')
        # print('GPU Memory Allocated: %d , Reserved: %d'%(torch.cuda.memory_allocated(),torch.cuda.memory_reserved()))
        dtype = torch.float64
        torchGPUSparseModel = SNS_Torch_Large(net, dt=dt,dtype=dtype)
        torchGPUSparseInput = torch.tensor([current],dtype=dtype,device='cpu')
        # print('CUDA Model Made')
        # print('GPU Memory Allocated: %d , Reserved: %d' % (torch.cuda.memory_allocated(), torch.cuda.memory_reserved()))
        for i in range(len(t)):
            print('%i Neurons Torch Sparse Step %i/%i'%(numNeurons[num],i+1,len(t)))
            stepStart = time.time()
            _ = torchGPUSparseModel.forward(torchGPUSparseInput)
            stepStop = time.time()
            torchGPUSparseTimes[num, i] = stepStop - stepStart
        del torchGPUSparseModel
        del torchGPUSparseInput
        del _
        # print('CUDA Models deleted')
        # print('GPU Memory Allocated: %d , Reserved: %d' % (torch.cuda.memory_allocated(), torch.cuda.memory_reserved()))
        torch.cuda.empty_cache()
        # print('CUDA cache cleared')
        # print('GPU Memory Allocated: %d , Reserved: %d' % (torch.cuda.memory_allocated(), torch.cuda.memory_reserved()))
        print('Finished Torch Sparse GPU. Running for %f sec' % (time.time() - globalStart))
    except:
        print('Sparse Cuda compilation failed. Running for %f sec' % (time.time() - globalStart))
        for i in range(len(t)):
            torchGPUTimes[num, i] = 0


    data = {'num_neurons': numNeurons,
            'numpy': npTimes,
            'torchCPU': torchCPUTimes,
            'torchGPU': torchGPUTimes,
            'torchGPUTransfer': torchGPUTransferTimes,
            'torchGPUSparse': torchGPUSparseTimes}
    pickle.dump(data, open('dataBackendTimesLessPts.p','wb'))
print('Finished test loop. Running for %f sec'%(time.time()-globalStart))
