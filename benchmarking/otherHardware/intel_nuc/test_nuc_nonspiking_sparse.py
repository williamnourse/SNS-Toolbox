import sys
import numpy as np
import torch
import time
import pickle

from sns_toolbox.neurons import NonSpikingNeuron
from sns_toolbox.connections import NonSpikingSynapse
from sns_toolbox.networks import Network

# Personal stuff to send an email once data collection is finished
# sys.path.extend(['/home/will'])
# from email_utils import send_email

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
numSamples = 50
numNeurons = np.geomspace(10,1000,num=numSamples)
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

    # Numpy
    npModel = net.compile(dt=dt,backend='numpy', device='cpu')
    npInput = np.array([current])
    for i in range(len(t)):
        print('Nonspiking Sparse. %i Neurons Numpy Step %i/%i'%(numNeurons[num],i+1,len(t)))
        stepStart = time.time()
        _ = npModel.forward(npInput)
        stepStop = time.time()
        npTimes[num,i] = stepStop-stepStart
    print('Finished Numpy. Running for %f sec' % (time.time() - globalStart))

    # Torch CPU
    torchCPUModel = net.compile(dt=dt,backend='torch', device='cpu')
    torchCPUInput = torch.tensor([current],device='cpu')
    for i in range(len(t)):
        print('Nonspiking Sparse. %i Neurons Torch CPU Step %i/%i'%(numNeurons[num],i+1,len(t)))
        stepStart = time.time()
        _ = torchCPUModel.forward(torchCPUInput)
        stepStop = time.time()
        torchCPUTimes[num, i] = stepStop - stepStart
    print('Finished Torch CPU. Running for %f sec' % (time.time() - globalStart))

    try:
        # Torch GPU
        # print('Before network created')
        # print('GPU Memory Allocated: %d , Reserved: %d'%(torch.cuda.memory_allocated(),torch.cuda.memory_reserved()))
        torchGPUModel = net.compile(dt=dt,backend='torch', device='cuda')
        torchGPUInput = torch.tensor([current],device='cuda')
        # print('CUDA Model Made')
        # print('GPU Memory Allocated: %d , Reserved: %d' % (torch.cuda.memory_allocated(), torch.cuda.memory_reserved()))
        for i in range(len(t)):
            print('Nonspiking Sparse. %i Neurons Torch GPU Step %i/%i'%(numNeurons[num],i+1,len(t)))
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
        # Torch Sparse CPU
        # print('Before network created')
        # print('GPU Memory Allocated: %d , Reserved: %d'%(torch.cuda.memory_allocated(),torch.cuda.memory_reserved()))
        torchGPUSparseModel = net.compile(dt=dt,backend='sparse', device='cpu')
        torchGPUSparseInput = torch.tensor([current],device='cpu')
        # print('CUDA Model Made')
        # print('GPU Memory Allocated: %d , Reserved: %d' % (torch.cuda.memory_allocated(), torch.cuda.memory_reserved()))
        for i in range(len(t)):
            print('Nonspiking Sparse. %i Neurons Torch Sparse CPU Step %i/%i'%(numNeurons[num],i+1,len(t)))
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
            print('Nonspiking Sparse. %i Neurons Torch Sparse GPU Step %i/%i'%(numNeurons[num],i+1,len(t)))
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

    # Numpy
    manModel = net.compile(dt=dt,backend='iterative', device='cpu')
    manInput = np.array([current])
    for i in range(len(t)):
        print('Nonspiking Sparse. %i Neurons Manual Step %i/%i' % (numNeurons[num], i + 1, len(t)))
        stepStart = time.time()
        _ = manModel.forward(npInput)
        stepStop = time.time()
        manualTimes[num, i] = stepStop - stepStart
    print('Finished Manual. Running for %f sec' % (time.time() - globalStart))

    data = {'shape': numNeurons,
            'numpy': npTimes,
            'torchCPU': torchCPUTimes,
            'torchGPU': torchGPUTimes,
            'sparseCPU': sparseCPUTimes,
            'sparseGPU': sparseGPUTimes,
            'manual': manualTimes}

    pickle.dump(data, open('dataNUCNonspikingSparse.p', 'wb'))

# send_email('wrn13@case.edu')
print('Finished test loop. Running for %f sec'%(time.time()-globalStart))
