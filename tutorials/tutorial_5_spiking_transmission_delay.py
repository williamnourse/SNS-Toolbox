"""
Let's build an example spiking network which showcases connections with spiking transmission delay
William Nourse
December 10th 2021
"""

import numpy as np
import matplotlib.pyplot as plt
import torch

from sns_toolbox.networks import Network
from sns_toolbox.neurons import SpikingNeuron
from sns_toolbox.connections import SpikingSynapse
from sns_toolbox.plot_utilities import spike_raster_plot
from sns_toolbox.renderer import render

"""Define our types"""
neuron_type = SpikingNeuron()
synapse_type_d0 = SpikingSynapse(transmission_delay=0)   # Transmission delay of 0 dt
synapse_type_d5 = SpikingSynapse(transmission_delay=5)   # Transmission delay of 5 dt
synapse_type_d10 = SpikingSynapse(transmission_delay=10)   # Transmission delay of 10 dt
synapse_type_d20 = SpikingSynapse(transmission_delay=20)   # Transmission delay of 20 dt

"""Create our network"""
net = Network(name='Tutorial 5 Network')

net.add_neuron(neuron_type, name='Source', color='blue')
net.add_neuron(neuron_type, name='D0', color='orange')
net.add_neuron(neuron_type, name='D5', color='green')
net.add_neuron(neuron_type, name='D10', color='red')
net.add_neuron(neuron_type, name='D20', color='purple')

net.add_connection(synapse_type_d0, 'Source', 'D0')
net.add_connection(synapse_type_d5, 'Source', 'D5')
net.add_connection(synapse_type_d10, 'Source', 'D10')
net.add_connection(synapse_type_d20, 'Source', 'D20')

net.add_output('Source', name='OSS', spiking=True)
net.add_output('D0', name='O0S', spiking=True)
net.add_output('D5', name='O5S', spiking=True)
net.add_output('D10', name='O10S', spiking=True)
net.add_output('D20', name='O20S', spiking=True)

net.add_input('Source')

render(net)

"""Simulate the network"""
dt = 0.01
t_max = 10

t = np.arange(0, t_max, dt)
inputs = torch.zeros([len(t), net.get_num_inputs()])          # getNumInputs() gets the number of input nodes in a network
inputs[0:100] = 20.0
data = torch.zeros([len(t), net.get_num_outputs_actual()])    # getNumOutputsActual gets the number of accessible output
                                                            # nodes in a network (since this net has populations, each
                                                            # population has n output nodes)
# Compile to numpy
model = net.compile(backend='sparse', dt=dt, debug=False)

# Run for all steps
for i in range(len(t)):
    data[i,:] = model(inputs[i,:])
data = data.transpose(0,1)

"""Plotting the results"""
plt.figure()
plt.subplot(2,1,1)
plt.plot(t,inputs,color='black')
plt.title('Input Stimulus')
plt.ylabel('Current (nA)')
plt.subplot(2,1,2)
spike_raster_plot(t,data[:][:],colors=['blue','orange','green','red','purple'])
plt.title('Network Spike Times')
plt.ylabel('Neuron')
plt.xlabel('t (ms)')


plt.show()