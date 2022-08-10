"""
Demo saving a compiled network, then loading and executing.
"""
import pickle

from sns_toolbox.design.neurons import NonSpikingNeuron, NonSpikingNeuronWithPersistentSodiumChannel, SpikingNeuron
from sns_toolbox.design.connections import NonSpikingSynapse, SpikingSynapse, ElectricalSynapse
from sns_toolbox.design.networks import Network

from sns_toolbox.simulate.backends import SNS_Numpy

import numpy as np
import matplotlib.pyplot as plt

neuron_non_spike = NonSpikingNeuron()
neuron_spike = SpikingNeuron()
neuron_gated = NonSpikingNeuronWithPersistentSodiumChannel()

synapse_non_spike = NonSpikingSynapse()
synapse_spike = SpikingSynapse()
synapse_spike_delay = SpikingSynapse(transmission_delay=5)
synapse_elect = ElectricalSynapse(conductance=1)
synapse_elect_rect = ElectricalSynapse(conductance=1,rect=True)

net = Network('Tutorial 9 Network')
net.add_neuron(neuron_non_spike,name='0',color='blue')
net.add_neuron(neuron_non_spike,name='1',color='orange')
net.add_neuron(neuron_spike,name='2',color='green')
net.add_neuron(neuron_spike,name='3',color='red')
net.add_neuron(neuron_gated,name='4',color='purple')
net.add_neuron(neuron_non_spike,name='5',color='brown')
net.add_neuron(neuron_non_spike,name='6',color='pink',initial_value=5)
net.add_neuron(neuron_spike, name='7',color='gray')

net.add_input('0')
net.add_input('2')
net.add_input('4')

net.add_output('0')
net.add_output('1')
net.add_output('2')
net.add_output('3')
net.add_output('4')
net.add_output('5')
net.add_output('6')
net.add_output('7')

net.add_connection(synapse_non_spike,'0','1')
net.add_connection(synapse_spike,'2','3')
net.add_connection(synapse_elect,'0','5')
net.add_connection(synapse_elect_rect,'6','5')
net.add_connection(synapse_spike_delay,'2','3')

# net.render_graph(view=True)
dt = 0.01
t_max = 50

# Initialize a vector of timesteps
t = np.arange(0, t_max, dt)

# Initialize vectors which store the input to our network, and for data to be written to during simulation from outputs
inputs = np.zeros([len(t),net.get_num_inputs()])+5.0 # Input vector must be 2d, even if second dimension is 1
data_original = np.zeros([len(t),net.get_num_outputs_actual()])
data_load = np.zeros([len(t),net.get_num_outputs_actual()])

model_original = SNS_Numpy(net,dt=dt)
# model_original.save(filename='saveTest.p')
# model_load = SNS_Numpy('saveTest.p',dt=dt)

pickle.dump(model_original, open('test.p','wb'))
model_load = pickle.load(open('test.p','rb'))

"""Simulate the network"""
# At every step, apply the current input to a forward pass of the network and store the results in 'data'
for i in range(len(t)):
    data_original[i,:] = model_original.forward(inputs[i,:])
    data_load[i, :] = model_load.forward(inputs[i, :])
data_original = data_original.transpose()
data_load = data_load.transpose()

plt.figure()
plt.subplot(2,1,1)
plt.plot(t,data_original[:][0],label='0',color='C0')
plt.plot(t,data_original[:][1],label='1',color='C1')
plt.plot(t,data_original[:][2],label='2',color='C2')
plt.plot(t,data_original[:][3],label='3',color='C3')
plt.plot(t,data_original[:][4],label='4',color='C4')
plt.plot(t,data_original[:][5],label='5',color='C5')
plt.plot(t,data_original[:][6],label='6',color='C6')
plt.plot(t,data_original[:][7],label='7',color='C7')
plt.legend()

plt.subplot(2,1,2)
plt.plot(t,data_original[:][0] - data_load[:][0],label='0',color='C0')
plt.plot(t,data_original[:][1] - data_load[:][1],label='1',color='C1')
plt.plot(t,data_original[:][2] - data_load[:][2],label='2',color='C2')
plt.plot(t,data_original[:][3] - data_load[:][3],label='3',color='C3')
plt.plot(t,data_original[:][4] - data_load[:][4],label='4',color='C4')
plt.plot(t,data_original[:][5] - data_load[:][5],label='5',color='C5')
plt.plot(t,data_original[:][6] - data_load[:][6],label='6',color='C6')
plt.plot(t,data_original[:][7] - data_load[:][7],label='7',color='C7')
plt.legend()

plt.show()
