"""
Compare the simulation speed of the different backends
William Nourse
September 9 2021
The correct term is Babes, sir
"""

import numpy as np
import nengo
import time
import pickle

from sns_toolbox.neurons import SpikingNeuron
from sns_toolbox.connections import SpikingSynapse
from sns_toolbox.networks import Network

# Personal stuff to send an email once data collection is finished
import sys
sys.path.extend(['/home/will'])
from email_utils import send_email

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

class SpikingNeuronsWithSynapticInput(nengo.Process):
    def __init__(self, num_neurons, Cm, Gmax, DelE, tau, m, theta0=1, bias=None, R=1, **kwargs):
        self.num_neurons = num_neurons

        if bias is None:
            self.bias = np.zeros(num_neurons)
        elif len(bias) == 1:
            self.bias = np.zeros(num_neurons) + bias[0]
        else:
            self.bias = bias
        self.R = R
        self.Cm = Cm
        self.Gmax = Gmax
        self.DelE = DelE
        self.tau = tau
        self.m = m
        self.theta0 = theta0
        self.theta = np.zeros(num_neurons) + theta0
        self.spikes = np.zeros_like(self.theta)
        self.gsyn = np.zeros([self.num_neurons, self.num_neurons])

        super().__init__(default_size_in=num_neurons, default_size_out=num_neurons, **kwargs)

    def make_state(self, shape_in, shape_out, dt, dtype=None):
        u = np.zeros(self.num_neurons)

        return {'state': u}

    def make_step(self, shape_in, shape_out, dt, rng, state):
        bias = self.bias
        Cm = self.Cm
        R = self.R
        u = state['state']
        Gmax = self.Gmax
        DelE = self.DelE
        tau = self.tau
        m = self.m
        theta0 = self.theta0

        def step(t, x):
            self.gsyn *= 1 - dt*1000/tau
            # print(self.gsyn)
            i_syn = np.sum(self.gsyn * DelE, axis=1) - u * np.sum(self.gsyn, axis=1)
            # print(i_syn)
            u[:] += dt*1000/Cm*(-u + i_syn + bias)
            # print(u)
            self.theta += dt*1000/tau*(-self.theta + theta0 + m*u)
            self.spikes = np.sign(np.minimum(0, self.theta - u))
            self.gsyn = np.maximum(self.gsyn, (-self.spikes) * Gmax)
            u[:] = (u * (self.spikes + 1))
            # print(self.spikes)
            return u

        return step

"""
########################################################################################################################
TEST SETUP
"""
numSamples = 100
numNeurons = np.geomspace(10,5000,num=numSamples)
dt = 0.1
tMax = 10
t = np.arange(0,tMax,dt)
nengoTimes = np.zeros([numSamples, len(t)])


print('Finished test setup. Running for %f sec'%(time.time()-globalStart))

for num in range(numSamples):
    print('%i Neurons. Running for %f sec' % (numNeurons[num],time.time() - globalStart))
    net = Network()
    numIns = int(0.08 * numNeurons[num]) + 1
    numOuts = int(0.12 * numNeurons[num])
    numSyn = int(np.sqrt(numNeurons[num]))
    numRest = int(numNeurons[num]) - numIns - numSyn - numOuts
    num_neurons = int(numNeurons[num])
    net.add_population(spike, [numIns], name='ins')  # in puppy, num_inputs = 8% of network
    net.add_population(spikeBias, [numOuts], name='outs')  # in puppy, num_outputs = 12% of network
    net.add_population(spikeBias, [numSyn], name='connected')  # in puppy, numSyn = shape
    net.add_population(spikeBias, [numRest], name='rest')  # rest of the network
    net.add_connection(spikeExcite, 'connected', 'connected')
    net.add_input('ins')
    net.add_output('outs')

    # Numpy
    npModel = net.compile(dt=dt,backend='numpy', device='cpu')
    with nengo.Network() as model:
        layer = nengo.Node(
            SpikingNeuronsWithSynapticInput(num_neurons, npModel.c_m, npModel.g_max_spike, npModel.del_e, 1, -1,
                                            bias=[current], R=20))
        probe = nengo.Probe(layer)

    with nengo.Simulator(model, dt=dt / 1000) as sim:
        for i in range(len(t)):
            print('%i Neurons Nengo Step %i/%i' % (numNeurons[num], i + 1, len(t)))
            stepStart = time.time()
            sim.step()
            _ = sim.data[probe][i, :]
            stepStop = time.time()
            nengoTimes[num, i] = stepStop - stepStart
    print('Finished Nengo. Running for %f sec' % (time.time() - globalStart))

    data = {'shape': numNeurons,
            'nengo': nengoTimes}

    pickle.dump(data, open('../../backendSpeed/dataNengoTimesSpikingSparse.p', 'wb'))

send_email('wrn13@case.edu')
print('Finished test loop. Running for %f sec' % (time.time() - globalStart))
