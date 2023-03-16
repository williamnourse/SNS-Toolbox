import sys
import numpy as np
import torch
import time
import pickle

from sns_toolbox.neurons import NonSpikingNeuron, SpikingNeuron
from sns_toolbox.connections import NonSpikingSynapse, SpikingSynapse
from sns_toolbox.networks import Network

# Personal stuff to send an email once data collection is finished
# sys.path.extend(['/home/will'])
# from email_utils import send_email

"""
Nonspiking Dense
"""

"""
########################################################################################################################
NEURON AND SYNAPSE DEFINITIONS
"""
current = 10.0
globalStart = time.time()
neuron = NonSpikingNeuron(membrane_capacitance=10, name='neuron', color='aquamarine')
# neuronBias = NonSpikingNeuron(membrane_capacitance=10, name='bias', color='aquamarine', bias=current)
synapse = NonSpikingSynapse(max_conductance=0.5, reversal_potential=-60)
print('Finished type definition. Running for %f sec'%(time.time()-globalStart))

"""
########################################################################################################################
TEST SETUP
"""
numSamples = 100
numNeurons = np.geomspace(10,5000,num=numSamples)
dt = 0.1
tMax = 10
t = np.arange(0,tMax,dt)
npTimes = np.zeros([numSamples,len(t)])
torchCPUTimes = np.zeros([numSamples,len(t)])
torchGPUTimes = np.zeros([numSamples,len(t)])
sparseCPUTimes = np.zeros([numSamples,len(t)])
sparseGPUTimes = np.zeros([numSamples,len(t)])
manualTimes = np.zeros([numSamples, len(t)])

print('Finished test setup. Running for %f sec'%(time.time()-globalStart))

for num in range(numSamples):
    print('%i Neurons. Running for %f sec' % (numNeurons[num],time.time() - globalStart))
    print('Sample %i/%i' % (num+1, numSamples))
    net = Network()
    num_neurons = int(numNeurons[num])
    net.add_population(neuron, [num_neurons], name='connected')  # in puppy, num_inputs = 8% of network
    net.add_connection(synapse, 'connected', 'connected')
    net.add_input('connected')
    net.add_output('connected')

    try:
        # Torch Sparse CPU
        # print('Before network created')
        # print('GPU Memory Allocated: %d , Reserved: %d'%(torch.cuda.memory_allocated(),torch.cuda.memory_reserved()))
        torchGPUSparseModel = net.compile(dt=dt,backend='sparse', device='cpu')
        torchGPUSparseInput = torch.tensor([current],device='cpu')
        # print('CUDA Model Made')
        # print('GPU Memory Allocated: %d , Reserved: %d' % (torch.cuda.memory_allocated(), torch.cuda.memory_reserved()))
        for i in range(len(t)):
            print('Nonspiking Dense: %i Neurons Torch Sparse Step %i/%i'%(numNeurons[num],i+1,len(t)))
            stepStart = time.time()
            _ = torchGPUSparseModel.forward(torchGPUSparseInput)
            stepStop = time.time()
            sparseCPUTimes[num, i] = stepStop - stepStart
        del torchGPUSparseModel
        del torchGPUSparseInput
        del _
        # print('CUDA Models deleted')
        # print('GPU Memory Allocated: %d , Reserved: %d' % (torch.cuda.memory_allocated(), torch.cuda.memory_reserved()))
        torch.cuda.empty_cache()
        # print('CUDA cache cleared')
        # print('GPU Memory Allocated: %d , Reserved: %d' % (torch.cuda.memory_allocated(), torch.cuda.memory_reserved()))
        print('Finished Torch Sparse CPU. Running for %f sec' % (time.time() - globalStart))
    except:
        print('Sparse CPU compilation failed. Running for %f sec' % (time.time() - globalStart))
        for i in range(len(t)):
            sparseCPUTimes[num, i] = 0

    try:
        # Torch Sparse GPU
        # print('Before network created')
        # print('GPU Memory Allocated: %d , Reserved: %d'%(torch.cuda.memory_allocated(),torch.cuda.memory_reserved()))
        torchGPUSparseModel = net.compile(dt=dt,backend='sparse', device='cuda')
        torchGPUSparseInput = torch.tensor([current],device='cuda')
        # print('CUDA Model Made')
        # print('GPU Memory Allocated: %d , Reserved: %d' % (torch.cuda.memory_allocated(), torch.cuda.memory_reserved()))
        for i in range(len(t)):
            print('Nonspiking Dense: %i Neurons Torch Sparse Step %i/%i'%(numNeurons[num],i+1,len(t)))
            stepStart = time.time()
            _ = torchGPUSparseModel.forward(torchGPUSparseInput)
            stepStop = time.time()
            sparseGPUTimes[num, i] = stepStop - stepStart
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
            sparseGPUTimes[num, i] = 0


    data = {'shape': numNeurons,
            'sparseCPU': sparseCPUTimes,
            'sparseGPU': sparseGPUTimes}

    pickle.dump(data, open('dataNUCTimesSparseNonspikingDenseLarger.p', 'wb'))

"""
Nonspiking Sparse
"""
"""
########################################################################################################################
NEURON AND SYNAPSE DEFINITIONS
"""
current = 10.0
globalStart = time.time()
neuron = NonSpikingNeuron(membrane_capacitance=10, name='neuron', color='aquamarine')
neuronBias = NonSpikingNeuron(membrane_capacitance=10, name='bias', color='aquamarine', bias=current)
synapse = NonSpikingSynapse(max_conductance=0.5, reversal_potential=-60)
print('Finished type definition. Running for %f sec'%(time.time()-globalStart))

"""
########################################################################################################################
TEST SETUP
"""
numSamples = 100
numNeurons = np.geomspace(10,5000,num=numSamples)
dt = 0.1
tMax = 10
t = np.arange(0,tMax,dt)
npTimes = np.zeros([numSamples,len(t)])
torchCPUTimes = np.zeros([numSamples,len(t)])
torchGPUTimes = np.zeros([numSamples,len(t)])
sparseCPUTimes = np.zeros([numSamples,len(t)])
sparseGPUTimes = np.zeros([numSamples,len(t)])
manualTimes = np.zeros([numSamples, len(t)])

print('Finished test setup. Running for %f sec'%(time.time()-globalStart))

for num in range(numSamples):
    print('%i Neurons. Running for %f sec' % (numNeurons[num],time.time() - globalStart))
    net = Network()
    numIns = int(0.08 * numNeurons[num]) + 1
    numOuts = int(0.12 * numNeurons[num])
    numSyn = int(np.sqrt(numNeurons[num]))
    numRest = int(numNeurons[num]) - numIns - numSyn - numOuts
    net.add_population(neuron, [numIns], name='ins')  # in puppy, num_inputs = 8% of network
    net.add_population(neuronBias, [numOuts], name='outs')  # in puppy, num_outputs = 12% of network
    net.add_population(neuronBias, [numSyn], name='connected')  # in puppy, numSyn = shape
    net.add_population(neuronBias, [numRest], name='rest')  # rest of the network
    net.add_connection(synapse, 'connected', 'connected')
    net.add_input('ins')
    net.add_output('outs')

    try:
        # Torch Sparse CPU
        # print('Before network created')
        # print('GPU Memory Allocated: %d , Reserved: %d'%(torch.cuda.memory_allocated(),torch.cuda.memory_reserved()))
        torchGPUSparseModel = net.compile(dt=dt,backend='sparse', device='cpu')
        torchGPUSparseInput = torch.tensor([current],device='cpu')
        # print('CUDA Model Made')
        # print('GPU Memory Allocated: %d , Reserved: %d' % (torch.cuda.memory_allocated(), torch.cuda.memory_reserved()))
        for i in range(len(t)):
            print('Nonspiking Sparse: %i Neurons Torch Sparse Step %i/%i'%(numNeurons[num],i+1,len(t)))
            stepStart = time.time()
            _ = torchGPUSparseModel.forward(torchGPUSparseInput)
            stepStop = time.time()
            sparseCPUTimes[num, i] = stepStop - stepStart
        del torchGPUSparseModel
        del torchGPUSparseInput
        del _
        # print('CUDA Models deleted')
        # print('GPU Memory Allocated: %d , Reserved: %d' % (torch.cuda.memory_allocated(), torch.cuda.memory_reserved()))
        torch.cuda.empty_cache()
        # print('CUDA cache cleared')
        # print('GPU Memory Allocated: %d , Reserved: %d' % (torch.cuda.memory_allocated(), torch.cuda.memory_reserved()))
        print('Finished Torch Sparse CPU. Running for %f sec' % (time.time() - globalStart))
    except:
        print('Sparse CPU compilation failed. Running for %f sec' % (time.time() - globalStart))
        for i in range(len(t)):
            sparseCPUTimes[num, i] = 0

    try:
        # Torch Sparse GPU
        # print('Before network created')
        # print('GPU Memory Allocated: %d , Reserved: %d'%(torch.cuda.memory_allocated(),torch.cuda.memory_reserved()))
        torchGPUSparseModel = net.compile(dt=dt,backend='sparse', device='cuda')
        torchGPUSparseInput = torch.tensor([current],device='cuda')
        # print('CUDA Model Made')
        # print('GPU Memory Allocated: %d , Reserved: %d' % (torch.cuda.memory_allocated(), torch.cuda.memory_reserved()))
        for i in range(len(t)):
            print('Nonspiking Sparse: %i Neurons Torch Sparse Step %i/%i'%(numNeurons[num],i+1,len(t)))
            stepStart = time.time()
            _ = torchGPUSparseModel.forward(torchGPUSparseInput)
            stepStop = time.time()
            sparseGPUTimes[num, i] = stepStop - stepStart
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
            sparseGPUTimes[num, i] = 0

data = {'shape': numNeurons,
            'sparseCPU': sparseCPUTimes,
            'sparseGPU': sparseGPUTimes}

pickle.dump(data, open('dataNUCTimesSparseNonspikingSparseLarger.p', 'wb'))

"""
Spiking Dense
"""
"""
########################################################################################################################
NEURON AND SYNAPSE DEFINITIONS
"""
current = 10.0
globalStart = time.time()
spike = SpikingNeuron(name='m<0', threshold_proportionality_constant=-1, color='aquamarine')
spikeExcite = SpikingSynapse(name='Excitatory Spiking')
print('Finished type definition. Running for %f sec'%(time.time()-globalStart))

"""
########################################################################################################################
TEST SETUP
"""
numSamples = 100
numNeurons = np.geomspace(10,5000,num=numSamples)
dt = 0.1
tMax = 10
t = np.arange(0,tMax,dt)
npTimes = np.zeros([numSamples,len(t)])
torchCPUTimes = np.zeros([numSamples,len(t)])
torchGPUTimes = np.zeros([numSamples,len(t)])
sparseCPUTimes = np.zeros([numSamples,len(t)])
sparseGPUTimes = np.zeros([numSamples,len(t)])
manualTimes = np.zeros([numSamples, len(t)])

print('Finished test setup. Running for %f sec'%(time.time()-globalStart))

for num in range(numSamples):
    print('%i Neurons. Running for %f sec' % (numNeurons[num],time.time() - globalStart))
    net = Network()
    num_neurons = int(numNeurons[num])
    net.add_population(spike, [num_neurons], name='connected')  # in puppy, num_inputs = 8% of network
    net.add_connection(spikeExcite, 'connected', 'connected')
    net.add_input('connected')
    net.add_output('connected')

    try:
        # Torch Sparse CPU
        # print('Before network created')
        # print('GPU Memory Allocated: %d , Reserved: %d'%(torch.cuda.memory_allocated(),torch.cuda.memory_reserved()))
        torchGPUSparseModel = net.compile(dt=dt,backend='sparse', device='cpu')
        torchGPUSparseInput = torch.tensor([current],device='cpu')
        # print('CUDA Model Made')
        # print('GPU Memory Allocated: %d , Reserved: %d' % (torch.cuda.memory_allocated(), torch.cuda.memory_reserved()))
        for i in range(len(t)):
            print('Spiking Dense: %i Neurons Torch Sparse Step %i/%i'%(numNeurons[num],i+1,len(t)))
            stepStart = time.time()
            _ = torchGPUSparseModel.forward(torchGPUSparseInput)
            stepStop = time.time()
            sparseCPUTimes[num, i] = stepStop - stepStart
        del torchGPUSparseModel
        del torchGPUSparseInput
        del _
        # print('CUDA Models deleted')
        # print('GPU Memory Allocated: %d , Reserved: %d' % (torch.cuda.memory_allocated(), torch.cuda.memory_reserved()))
        torch.cuda.empty_cache()
        # print('CUDA cache cleared')
        # print('GPU Memory Allocated: %d , Reserved: %d' % (torch.cuda.memory_allocated(), torch.cuda.memory_reserved()))
        print('Finished Torch Sparse CPU. Running for %f sec' % (time.time() - globalStart))
    except:
        print('Sparse CPU compilation failed. Running for %f sec' % (time.time() - globalStart))
        for i in range(len(t)):
            sparseCPUTimes[num, i] = 0

    try:
        # Torch Sparse GPU
        # print('Before network created')
        # print('GPU Memory Allocated: %d , Reserved: %d'%(torch.cuda.memory_allocated(),torch.cuda.memory_reserved()))
        torchGPUSparseModel = net.compile(dt=dt,backend='sparse', device='cuda')
        torchGPUSparseInput = torch.tensor([current],device='cuda')
        # print('CUDA Model Made')
        # print('GPU Memory Allocated: %d , Reserved: %d' % (torch.cuda.memory_allocated(), torch.cuda.memory_reserved()))
        for i in range(len(t)):
            print('Spiking Dense: %i Neurons Torch Sparse Step %i/%i'%(numNeurons[num],i+1,len(t)))
            stepStart = time.time()
            _ = torchGPUSparseModel.forward(torchGPUSparseInput)
            stepStop = time.time()
            sparseGPUTimes[num, i] = stepStop - stepStart
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
            sparseGPUTimes[num, i] = 0

data = {'shape': numNeurons,
        'sparseCPU': sparseCPUTimes,
        'sparseGPU': sparseGPUTimes}

pickle.dump(data, open('dataNUCTimesSparseSpikingDenseLarger.p', 'wb'))

"""
Spiking Sparse
"""
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
numNeurons = np.geomspace(10,5000,num=numSamples)
dt = 0.1
tMax = 10
t = np.arange(0,tMax,dt)
npTimes = np.zeros([numSamples,len(t)])
torchCPUTimes = np.zeros([numSamples,len(t)])
torchGPUTimes = np.zeros([numSamples,len(t)])
sparseCPUTimes = np.zeros([numSamples,len(t)])
sparseGPUTimes = np.zeros([numSamples,len(t)])
manualTimes = np.zeros([numSamples, len(t)])

print('Finished test setup. Running for %f sec'%(time.time()-globalStart))

for num in range(numSamples):
    print('%i Neurons. Running for %f sec' % (numNeurons[num],time.time() - globalStart))
    net = Network()
    numIns = int(0.08 * numNeurons[num]) + 1
    numOuts = int(0.12 * numNeurons[num])
    numSyn = int(np.sqrt(numNeurons[num]))
    numRest = int(numNeurons[num]) - numIns - numSyn - numOuts
    net.add_population(spike, [numIns], name='ins')  # in puppy, num_inputs = 8% of network
    net.add_population(spikeBias, [numOuts], name='outs')  # in puppy, num_outputs = 12% of network
    net.add_population(spikeBias, [numSyn], name='connected')  # in puppy, numSyn = shape
    net.add_population(spikeBias, [numRest], name='rest')  # rest of the network
    net.add_connection(spikeExcite, 'connected', 'connected')
    net.add_input('ins')
    net.add_output('outs')

    try:
        # Torch Sparse CPU
        # print('Before network created')
        # print('GPU Memory Allocated: %d , Reserved: %d'%(torch.cuda.memory_allocated(),torch.cuda.memory_reserved()))
        torchGPUSparseModel = net.compile(dt=dt,backend='sparse', device='cpu')
        torchGPUSparseInput = torch.tensor([current],device='cpu')
        # print('CUDA Model Made')
        # print('GPU Memory Allocated: %d , Reserved: %d' % (torch.cuda.memory_allocated(), torch.cuda.memory_reserved()))
        for i in range(len(t)):
            print('Spiking Sparse: %i Neurons Torch Sparse Step %i/%i'%(numNeurons[num],i+1,len(t)))
            stepStart = time.time()
            _ = torchGPUSparseModel.forward(torchGPUSparseInput)
            stepStop = time.time()
            sparseCPUTimes[num, i] = stepStop - stepStart
        del torchGPUSparseModel
        del torchGPUSparseInput
        del _
        # print('CUDA Models deleted')
        # print('GPU Memory Allocated: %d , Reserved: %d' % (torch.cuda.memory_allocated(), torch.cuda.memory_reserved()))
        torch.cuda.empty_cache()
        # print('CUDA cache cleared')
        # print('GPU Memory Allocated: %d , Reserved: %d' % (torch.cuda.memory_allocated(), torch.cuda.memory_reserved()))
        print('Finished Torch Sparse CPU. Running for %f sec' % (time.time() - globalStart))
    except:
        print('Sparse CPU compilation failed. Running for %f sec' % (time.time() - globalStart))
        for i in range(len(t)):
            sparseCPUTimes[num, i] = 0

    try:
        # Torch Sparse GPU
        # print('Before network created')
        # print('GPU Memory Allocated: %d , Reserved: %d'%(torch.cuda.memory_allocated(),torch.cuda.memory_reserved()))
        torchGPUSparseModel = net.compile(dt=dt,backend='sparse', device='cuda')
        torchGPUSparseInput = torch.tensor([current],device='cuda')
        # print('CUDA Model Made')
        # print('GPU Memory Allocated: %d , Reserved: %d' % (torch.cuda.memory_allocated(), torch.cuda.memory_reserved()))
        for i in range(len(t)):
            print('Spiking Sparse: %i Neurons Torch Sparse Step %i/%i'%(numNeurons[num],i+1,len(t)))
            stepStart = time.time()
            _ = torchGPUSparseModel.forward(torchGPUSparseInput)
            stepStop = time.time()
            sparseGPUTimes[num, i] = stepStop - stepStart
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
            sparseGPUTimes[num, i] = 0

data = {'shape': numNeurons,
        'sparseCPU': sparseCPUTimes,
        'sparseGPU': sparseGPUTimes}

pickle.dump(data, open('dataNUCTimesSparseSpikingSparseLarger.p', 'wb'))

# send_email('wrn13@case.edu')
print('Finished test loop. Running for %f sec'%(time.time()-globalStart))
