import numpy as np
from ANNarchy import *
# import matplotlib.pyplot as plt
import time
import pickle

# Personal stuff to send an email once data collection is finished
import sys
sys.path.extend(['/home/will'])
from email_utils import send_email
"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
NEURON AND SYNAPSE DEFINITIONS
"""
globalStart = time.time()
setup(dt=0.1, paradigm='cuda')
SpikingNeuron = Neuron(
    parameters="""
        Cm = 5.0
        Gm = 1.0
        bias = 0.0
        tau = 5.0
        To = 1.0
        m = 1.0
        tau_inh = 1.0
        Esyn = -3.0
    """,
    equations="""
        Cm * dv/dt = -Gm * v + bias + g_inh * (Esyn-v) : init = 0.0
        tau * dT/dt = -T + To + m * v : init = 1.0
        tau_inh * dg_inh/dt = -g_inh
    """,
    spike = "v > T",
    reset = "v = 0"
)
SpikingSynapse = Synapse(
    parameters="""
        Gmax = 0.99
        Esyn = 5.0
    """,
    equations="""""",
    pre_spike="""
        g_target = Gmax : max = Gmax
    """
)

current = 0.5

print('Finished type definition. Running for %f sec'%(time.time()-globalStart))

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
TEST SETUP
"""
numSamples = 100
numNeurons = np.geomspace(10,5000,num=numSamples)

dt = 0.1
numSteps = 1000
annTimes = np.zeros([numSamples,numSteps])

print('Finished test setup. Running for %f sec' % (time.time()-globalStart))

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
TESTING LOOP
"""
for num in range(numSamples):
    print()
    print('Sample %i/%i : %i Neurons. Running for %f sec' % (num+1,numSamples,numNeurons[num],time.time() - globalStart))

    numIns = int(0.08 * numNeurons[num]) + 1  # in puppy, num_inputs = 8% of network
    numOuts = int(0.12 * numNeurons[num])  # in puppy, num_outputs = 12% of network
    numSyn = int(np.sqrt(numNeurons[num]))  # in puppy, numSyn = shape
    numRest = int(numNeurons[num]) - numIns - numSyn - numOuts  # rest of the network

    clear()
    bias = np.zeros(numIns) + current
    net = Network()

    ins = Population(geometry=numIns, neuron=SpikingNeuron)

    outs = Population(geometry=numOuts, neuron=SpikingNeuron)

    syn = Population(geometry=numSyn, neuron=SpikingNeuron)

    rest = Population(geometry=numRest, neuron=SpikingNeuron)

    connect_matrix = np.ones([numSyn, numSyn])
    proj = Projection(
        pre=syn,
        post=syn,
        target='inh',
        synapse=SpikingSynapse
    ).connect_from_matrix(connect_matrix)
    net.add([ins, outs, syn, rest, proj])
    net.compile()

    # net.add((ins, outs, syn, rest, connect))

    print('Finished network construction with %i neurons. Running for %f sec' % (
    numNeurons[num], time.time() - globalStart))

    for i in range(numSteps):
        print('%i Neurons ANNarchy Step %i/%i' % (numNeurons[num], i + 1, numSteps))
        stepStart = time.time()
        net.get(ins).bias = bias
        net.step()
        foo = net.get(outs).r
        stepStop = time.time()
        annTimes[num, i - 1] = stepStop - stepStart

    data = {'shape': numNeurons,
            'annarchy': annTimes}

    pickle.dump(data, open('dataANNarchyCUDATimesSpikingSparse.p', 'wb'))
send_email('wrn13@case.edu')
print('Finished test loop. Running for %f sec' % (time.time() - globalStart))
