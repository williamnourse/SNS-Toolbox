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
from sns_toolbox.simulate.backends import SNS_Torch, SNS_Torch_Large

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

numSamples = 10
numNeurons = np.logspace(1,3,num=numSamples)
# num_neurons = np.linspace(10000,100000,num=numSamples)
dt = 0.01

print('Finished test setup. Running for %f sec'%(time.time()-globalStart))

for num in range(numSamples):
    print('\n%i Neurons. Running for %f sec' % (numNeurons[num],time.time() - globalStart))
    net = Network()
    numIns = int(0.08*numNeurons[num])+1
    numOuts = int(0.12 * numNeurons[num])
    numSyn = int(np.sqrt(numNeurons[num]))
    numRest = int(numNeurons[num]) - numIns - numSyn - numOuts
    net.add_population(spike, numIns, name='ins')  # in puppy, num_inputs = 8% of network
    net.add_population(spikeBias, numOuts, name='outs')  # in puppy, num_outputs = 12% of network
    net.add_population(spikeBias, numSyn, name='connected')  # in puppy, numSyn = num_neurons
    net.add_population(spikeBias, numRest, name='rest')  # rest of the network
    net.add_synapse(spikeExcite, 'connected', 'connected')
    net.add_input('Input')
    net.addInputConnection(1.0,'Input','ins')
    net.add_output('outs')

    # Torch GPU
    # print('Before network created')
    # print('GPU Memory Allocated: %d , Reserved: %d'%(torch.cuda.memory_allocated(),torch.cuda.memory_reserved()))
    # try:
    dtype = torch.float32
    torchGPUModel = SNS_Torch_Large(net, dt=dt,dtype=dtype)#, device='cuda')
    torchGPUInput = torch.tensor([current],dtype=dtype,device='cpu')
    # print('CUDA Model Made')
    print('GPU Memory Allocated: %d , Reserved: %d' % (torch.cuda.memory_allocated(), torch.cuda.memory_reserved()))
    _ = torchGPUModel.forward(torchGPUInput)
    del torchGPUModel
    del torchGPUInput
    del _
    # print('CUDA Models deleted')
    # print('GPU Memory Allocated: %d , Reserved: %d' % (torch.cuda.memory_allocated(), torch.cuda.memory_reserved()))
    torch.cuda.empty_cache()
    # print('CUDA cache cleared')
    # print('GPU Memory Allocated: %d , Reserved: %d' % (torch.cuda.memory_allocated(), torch.cuda.memory_reserved()))
    print('Finished Torch GPU. Running for %f sec' % (time.time() - globalStart))
    # except:
    #     print('CUDA Compilation Failed')
    #     break
print('Finished test loop. Running for %f sec'%(time.time()-globalStart))
