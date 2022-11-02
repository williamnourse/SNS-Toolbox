import sys
import numpy as np
import nengo
import time
import pickle

from sns_toolbox.neurons import NonSpikingNeuron
from sns_toolbox.connections import NonSpikingSynapse
from sns_toolbox.networks import Network

# Personal stuff to send an email once data collection is finished
sys.path.extend(['/home/will'])
from email_utils import send_email

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

class NonSpikingNeuronsWithSynapticInput(nengo.Process):
    def __init__(self, num_neurons, Cm, Gmax, DelE, bias=None, R=1, **kwargs):
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

        super().__init__(default_size_in=num_neurons, default_size_out=num_neurons, **kwargs)

    def make_state(self, shape_in, shape_out, dt, dtype=None):
        return {'state': np.zeros(self.num_neurons)}

    def make_step(self, shape_in, shape_out, dt, rng, state):
        bias = self.bias
        Cm = self.Cm
        R = self.R
        u = state['state']
        Gmax = self.Gmax
        DelE = self.DelE

        def step(t, x):
            g_syn = np.maximum(0, np.minimum(self.Gmax * (u / R),
                                             self.Gmax))

            i_syn = np.sum(g_syn * DelE, axis=1) - u * np.sum(g_syn, axis=1)
            u[:] += dt*1000/Cm*(-u + i_syn + bias)
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
nengoTimes = np.zeros([numSamples,len(t)])

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

    # Numpy
    npModel = net.compile(dt=dt,backend='numpy', device='cpu')
    with nengo.Network() as model:
        layer = nengo.Node(NonSpikingNeuronsWithSynapticInput(num_neurons, npModel.c_m, npModel.g_max_non, npModel.del_e, bias=[current], R=20))
        probe = nengo.Probe(layer)

    with nengo.Simulator(model, dt=dt/1000) as sim:
        for i in range(len(t)):
            print('%i Neurons Nengo Step %i/%i' % (numNeurons[num], i + 1, len(t)))
            stepStart = time.time()
            sim.step()
            _ = sim.data[probe][-1, :]
            stepStop = time.time()
            nengoTimes[num,i] = stepStop-stepStart
    print('Finished Nengo. Running for %f sec' % (time.time() - globalStart))

    data = {'shape': numNeurons,
            'nengo': nengoTimes}

    pickle.dump(data, open('../../backendSpeed/dataNengoTimesNonspikingDense.p', 'wb'))

send_email('wrn13@case.edu')
print('Finished test loop. Running for %f sec'%(time.time()-globalStart))
