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
I : 1
Isyn : 1
Ibias : 1
'''
tau = 10*ms

current = 10.0

eqs_synapse = '''
Isyn_post = Gmax*clip(v_pre/R, 0, 1)*(DelE - v_post) : 1 (summed)
'''
R = 20.0
DelE = -3*R
Gmax = 0.5

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
    print()
    print('%i Neurons. Running for %f sec' % (numNeurons[num],time.time() - globalStart))

    numIns = int(0.08 * numNeurons[num]) + 1                    # in puppy, num_inputs = 8% of network
    numOuts = int(0.12 * numNeurons[num])                       # in puppy, num_outputs = 12% of network
    numSyn = int(np.sqrt(numNeurons[num]))                      # in puppy, numSyn = shape
    numRest = int(numNeurons[num]) - numIns - numSyn - numOuts  # rest of the network

    start_scope()

    # net = Network()

    ins = NeuronGroup(numIns, eqs_neuron, method='euler')
    ins.I = np.zeros(numIns)+current

    outs = NeuronGroup(numOuts, eqs_neuron, method='euler')
    outs.Ibias =np.zeros(numOuts)+current

    syn = NeuronGroup(numSyn, eqs_neuron, method='euler')
    syn.Ibias = np.zeros(numSyn) + current

    rest = NeuronGroup(numRest, eqs_neuron, method='euler')
    rest.Ibias = np.zeros(numRest) + current

    connect = Synapses(syn,syn, eqs_synapse)
    connect_matrix = np.ones([numSyn, numSyn])
    sources, targets = connect_matrix.nonzero()
    connect.connect(i=sources, j=targets)

    # net.add((ins, outs, syn, rest, connect))

    print('Finished network construction with %i neurons. Running for %f sec' % (numNeurons[num], time.time() - globalStart))

    for i in range(numSteps):
        print('%i Neurons Brian Step %i/%i'%(numNeurons[num],i+1,numSteps))
        stepStart = time.time()
        run(dt*ms)
        _ = outs.v
        stepStop = time.time()
        if i > 0:
            brianTimes[num,i-1] = stepStop-stepStart

    data = {'shape': numNeurons,
            'brian': brianTimes}

    pickle.dump(data, open('../backendSpeed/dataBrianTimesNonspikingSparse.p', 'wb'))
print('Finished test loop. Running for %f sec' % (time.time() - globalStart))

# tmax = 50 # ms
# t = np.arange(0,tmax,dt)
# data = np.zeros([len(t),num_neurons])
# # G.v = 'rand()'
# data[0,:] = pop.v
# for i in range(1, len(data)):
#     # print(G.v[0])
#     print(i)
#     run(0.1*ms)
#     data[i,:] = pop.v
#     # print(G.v[0])
# data = data.transpose()
# plt.figure()
# for i in range(num_neurons):
#     plt.plot(t,data[i,:])
# plt.show()
