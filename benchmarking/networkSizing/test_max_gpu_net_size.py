"""
Let's go hunting for what the max network size our GeForce RTX 2060 can sustain is
William Nourse
September 10 2021
Give Gus his twitter handle
"""
import numpy as np
import torch
import time

from sns_toolbox.design.neurons import SpikingNeuron
from sns_toolbox.design.connections import SpikingSynapse
from sns_toolbox.design.networks import Network
from sns_toolbox.simulate.backends import SNS_Torch

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
numSamples = 3
numNeurons = np.logspace(1,4,num=numSamples)
# numNeurons = np.linspace(7224,7244,num=numSamples)
dt = 0.01

print('Finished test setup. Running for %f sec'%(time.time()-globalStart))

for num in range(numSamples):
    print('\n%i Neurons. Running for %f sec' % (numNeurons[num],time.time() - globalStart))
    net = Network()
    net.addPopulation(spikeL0,int(numNeurons[num]),name='self')
    net.addSynapse(spikeExcite,'self','self')
    net.addInput('Input')
    net.addInputConnection(1.0,'Input','self')
    net.addOutput('self')

    # Torch GPU
    # print('Before network created')
    # print('GPU Memory Allocated: %d , Reserved: %d'%(torch.cuda.memory_allocated(),torch.cuda.memory_reserved()))
    torchGPUModel = SNS_Torch(net, dt=dt, device='cuda')
    torchGPUInput = torch.tensor([current],dtype=torch.float64,device='cuda')
    # print('CUDA Model Made')
    print('GPU Memory Allocated: %d , Reserved: %d' % (torch.cuda.memory_allocated(), torch.cuda.memory_reserved()))

    del torchGPUModel
    del torchGPUInput
    # print('CUDA Models deleted')
    # print('GPU Memory Allocated: %d , Reserved: %d' % (torch.cuda.memory_allocated(), torch.cuda.memory_reserved()))
    torch.cuda.empty_cache()
    # print('CUDA cache cleared')
    # print('GPU Memory Allocated: %d , Reserved: %d' % (torch.cuda.memory_allocated(), torch.cuda.memory_reserved()))
    print('Finished Torch GPU. Running for %f sec' % (time.time() - globalStart))
print('Finished test loop. Running for %f sec'%(time.time()-globalStart))
