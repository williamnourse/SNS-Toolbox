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
from sns_toolbox.simulate.backends import SNS_Numpy, SNS_Torch

"""
########################################################################################################################
NEURON AND SYNAPSE DEFINITIONS
"""

globalStart = time.time()
spikeL0 = SpikingNeuron(name='m<0',thresholdProportionalityConstant=-1,color='aquamarine')
spikeExcite = SpikingSynapse(name='Excitatory Spiking')
print('Finished type definition. Running for %f sec'%(time.time()-globalStart))

"""
########################################################################################################################
TEST SETUP
"""

current = 10.0
numSamples = 5
numNeurons = np.logspace(1,2,num=numSamples)
dt = 0.01
tMax = 100
t = np.arange(0,tMax,dt)
npTimes = np.zeros([numSamples,len(t)])
torchCPUTimes = np.zeros([numSamples,len(t)])
torchGPUTimes = np.zeros([numSamples,len(t)])
torchGPUTransferTimes = np.zeros([numSamples,len(t)])

print('Finished test setup. Running for %f sec'%(time.time()-globalStart))

for num in range(numSamples):
    print('%i Neurons. Running for %f sec' % (numNeurons[num],time.time() - globalStart))
    net = Network()
    net.addPopulation(spikeL0,int(numNeurons[num]),name='self')
    net.addSynapse(spikeExcite,'self','self')
    net.addInput('Input')
    net.addInputConnection(1.0,'Input','self')
    net.addOutput('self')

    # Numpy
    npModel = SNS_Numpy(net,dt=dt)
    npInput = np.array([current])
    for i in range(len(t)):
        stepStart = time.time()
        _ = npModel.forward(npInput)
        stepStop = time.time()
        npTimes[num,i] = stepStop-stepStart
    print('Finished Numpy. Running for %f sec' % (time.time() - globalStart))

    # Torch CPU
    torchCPUModel = SNS_Torch(net,dt=dt,device='cpu')
    torchCPUInput = torch.tensor([current],dtype=torch.float64,device='cpu')
    for i in range(len(t)):
        stepStart = time.time()
        _ = torchCPUModel.forward(torchCPUInput)
        stepStop = time.time()
        torchCPUTimes[num, i] = stepStop - stepStart
    print('Finished Torch CPU. Running for %f sec' % (time.time() - globalStart))

    # Torch GPU
    # print('Before network created')
    # print('GPU Memory Allocated: %d , Reserved: %d'%(torch.cuda.memory_allocated(),torch.cuda.memory_reserved()))
    torchGPUModel = SNS_Torch(net, dt=dt, device='cuda')
    torchGPUInput = torch.tensor([current],dtype=torch.float64,device='cuda')
    # print('CUDA Model Made')
    # print('GPU Memory Allocated: %d , Reserved: %d' % (torch.cuda.memory_allocated(), torch.cuda.memory_reserved()))
    for i in range(len(t)):
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

    # Torch GPU with memory transfer at each step
    # print('Before network created')
    # print('GPU Memory Allocated: %d , Reserved: %d'%(torch.cuda.memory_allocated(),torch.cuda.memory_reserved()))
    torchGPUTransferModel = SNS_Torch(net, dt=dt, device='cuda')
    torchGPUTransferInput = torch.tensor([current], dtype=torch.float64, device='cpu')
    # print('CUDA Model Made')
    # print('GPU Memory Allocated: %d , Reserved: %d' % (torch.cuda.memory_allocated(), torch.cuda.memory_reserved()))
    for i in range(len(t)):
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

data = {'numNeurons': numNeurons,
        'numpy': npTimes,
        'torchCPU': torchCPUTimes,
        'torchGPU': torchGPUTimes,
        'torchGPUTransfer': torchGPUTransferTimes}
pickle.dump(data, open('dataBackendTimes.p','wb'))
print('Finished test loop. Running for %f sec'%(time.time()-globalStart))
