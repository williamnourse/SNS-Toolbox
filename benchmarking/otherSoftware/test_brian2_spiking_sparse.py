import matplotlib.pyplot as plt
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
    print()
    print('%i Neurons. Running for %f sec' % (numNeurons[num],time.time() - globalStart))

    numIns = int(0.08 * numNeurons[num]) + 1                    # in puppy, num_inputs = 8% of network
    numOuts = int(0.12 * numNeurons[num])                       # in puppy, num_outputs = 12% of network
    numSyn = int(np.sqrt(numNeurons[num]))                      # in puppy, numSyn = shape
    numRest = int(numNeurons[num]) - numIns - numSyn - numOuts  # rest of the network

    start_scope()

    # net = Network()

    ins = NeuronGroup(numIns, eqs_neuron, threshold='v>a', reset='v=0', method='euler')
    ins.I = np.zeros(numIns)+current

    outs = NeuronGroup(numOuts, eqs_neuron, threshold='v>a', reset='v=0', method='euler')
    outs.Ibias =np.zeros(numOuts)+current

    syn = NeuronGroup(numSyn, eqs_neuron, threshold='v>a', reset='v=0', method='euler')
    syn.Ibias = np.zeros(numSyn) + current

    rest = NeuronGroup(numRest, eqs_neuron, threshold='v>a', reset='v=0', method='euler')
    rest.Ibias = np.zeros(numRest) + current

    connect = Synapses(syn,syn, eqs_synapse, on_pre='G = Gmax', method='euler')
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

    pickle.dump(data, open('../backendSpeed/dataBrianTimesSpikingSparse.p', 'wb'))
print('Finished test loop. Running for %f sec' % (time.time() - globalStart))

# start_scope()
# tmax = 10 # ms
# t = np.arange(0,tmax,dt)
# data = np.zeros([len(t),2])
# synData = np.zeros(len(t))
# pop = NeuronGroup(2, eqs_neuron, threshold='v>a', reset='v=0', method='euler')
# pop.I = [2,0]
# pop.a = [1,1]
# syn = Synapses(pop,pop,eqs_synapse, on_pre='G = Gmax',method='euler')#,dt=defaultclock.dt)
# syn.connect(i=0,j=1)
# # G.v = 'rand()'
# data[0,:] = pop.v
# synData[0] = syn.G[0,1]
# for i in range(1, len(data)):
#     # print(G.v[0])
#     print(i)
#     run(0.1*ms)
#     data[i,:] = pop.v
#     synData[i] = syn.G[0, 1]
#     # print(G.v[0])
# print(syn.dt)
# data = data.transpose()
# plt.figure()
# plt.subplot(2,1,1)
# for i in range(2):
#     plt.plot(t,data[i,:])
# plt.subplot(2,1,2)
# plt.plot(t,synData)
#
# plt.show()
