import numpy as np
from brian2 import *
# import matplotlib.pyplot as plt
import time
import pickle

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
NEURON AND SYNAPSE DEFINITIONS
"""
globalStart = time.time()
eqs_neuron = '''
dv/dt = (-v+I+Isyn+Ibias)/tau : 1
da/dt = (-a + 1 + m*v)/tau : 1
I : 1
Isyn : 1
Ibias : 1
'''
tau = 5*ms
m = 0

current = 10.0

eqs_synapse = '''
Isyn_post = G*(DelE - v_post) : 1 (summed)
dG/dt = -G/tauG : 1 (clock-driven)
'''
tauG = 1*ms
R = 20.0
DelE = 194
Gmax = 1

print('Finished type definition. Running for %f sec'%(time.time()-globalStart))

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
TEST SETUP
"""
numSamples = 100
numNeurons = np.geomspace(10,5000,num=numSamples)

dt = 0.1
numSteps = 1001
defaultclock.dt = dt*ms
brianTimes = np.zeros([numSamples,numSteps-1])

print('Finished test setup. Running for %f sec' % (time.time()-globalStart))

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
TESTING LOOP
"""
for num in range(numSamples):
    num_neurons = int(numNeurons[num])
    print('%i Neurons. Running for %f sec' % (num_neurons,time.time() - globalStart))

    start_scope()

    # net = Network()

    nrns = NeuronGroup(num_neurons, eqs_neuron, threshold='v>a', reset='v=0', method='euler')
    nrns.I = np.zeros(num_neurons)+current

    connect = Synapses(nrns, nrns, eqs_synapse, on_pre='G = Gmax', method='euler')
    connect_matrix = np.ones([num_neurons, num_neurons])
    sources, targets = connect_matrix.nonzero()
    connect.connect(i=sources, j=targets)

    # net.add((ins, outs, syn, rest, connect))

    print('Finished network construction with %i neurons. Running for %f sec' % (num_neurons, time.time() - globalStart))

    for i in range(numSteps):
        print('Sample %i/%i' % (num + 1, numSamples))
        print('%i Neurons Brian Step %i/%i'%(num_neurons,i+1,numSteps))
        stepStart = time.time()
        run(dt*ms)
        _ = nrns.v
        stepStop = time.time()
        if i > 0:
            brianTimes[num,i-1] = stepStop-stepStart

    data = {'shape': numNeurons,
            'brian': brianTimes}

    pickle.dump(data, open('../backendSpeed/dataBrianTimesSpikingDense.p', 'wb'))
print('Finished test loop. Running for %f sec' % (time.time() - globalStart))