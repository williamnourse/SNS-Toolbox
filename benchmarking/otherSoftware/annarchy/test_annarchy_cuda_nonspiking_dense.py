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
NonSpikingNeuron = Neuron(
    parameters="""
        Cm = 5.0
        Gm = 1.0
        bias = 0.0
        Esyn = -60.0
    """,
    equations="""
        Cm * dv/dt = -Gm * v + bias + sum(inh)*(Esyn-v)
        r = v
    """
)
NonSpikingSynapse = Synapse(
    parameters="""
        Gmax = 0.5
        El = 0.0
        Eh = 20.0
    """,
    equations="""
        w = clip(Gmax * (pre.r-El)/(Eh-El), 0.0, Gmax)
    """,
    psp="""
        w
    """
)

current = 10.0

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
    clear()
    num_neurons = int(numNeurons[num])
    print('Sample %i/%i : %i Neurons. Running for %f sec' % (num+1,numSamples,num_neurons,time.time() - globalStart))

    net = Network()

    nrns = Population(geometry=num_neurons,neuron=NonSpikingNeuron)
    # nrns1 = Population(geometry=num_neurons,neuron=NonSpikingNeuron)
    connect_matrix = np.ones([num_neurons,num_neurons])
    proj = Projection(
        pre=nrns,
        post=nrns,
        target='inh',
        synapse=NonSpikingSynapse
    ).connect_from_matrix(connect_matrix)
    net.add([nrns,proj])
    net.compile()
    bias = np.zeros(num_neurons)+current

    # net.add((ins, outs, syn, rest, connect))

    print('Finished network construction with %i neurons. Running for %f sec' % (num_neurons, time.time() - globalStart))

    for i in range(numSteps):
        print('%i Neurons ANNarchy Step %i/%i'%(num_neurons,i+1,numSteps))
        stepStart = time.time()
        net.get(nrns).bias = bias
        net.step()
        foo = net.get(nrns).r
        stepStop = time.time()
        annTimes[num,i] = stepStop-stepStart

    data = {'shape': numNeurons,
            'annarchy': annTimes}

    pickle.dump(data, open('dataANNarchyCUDATimesNonspikingDense.p', 'wb'))
send_email('wrn13@case.edu')
print('Finished test loop. Running for %f sec' % (time.time() - globalStart))