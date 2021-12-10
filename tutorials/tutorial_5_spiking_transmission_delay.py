"""
Let's build an example spiking network which showcases synapses with spiking transmission delay
William Nourse
December 10th 2021
"""

import numpy as np
import matplotlib.pyplot as plt

from sns_toolbox.design.networks import Network
from sns_toolbox.design.neurons import SpikingNeuron
from sns_toolbox.design.connections import SpikingSynapse

from sns_toolbox.simulate.backends import SNS_Numpy
from sns_toolbox.simulate.plotting import spike_raster_plot

"""Define our types"""
neuron_type = SpikingNeuron()
synapse_type_d0 = SpikingSynapse(transmissionDelay=0)   # Transmission delay of 0 dt
synapse_type_d5 = SpikingSynapse(transmissionDelay=5)   # Transmission delay of 5 dt
synapse_type_d10 = SpikingSynapse(transmissionDelay=10)   # Transmission delay of 10 dt
synapse_type_d20 = SpikingSynapse(transmissionDelay=20)   # Transmission delay of 20 dt

"""Create our network"""
net = Network(name='Tutorial 5 Network')

net.addNeuron(neuron_type,name='Source',color='blue')
net.addNeuron(neuron_type,name='D0',color='orange')
net.addNeuron(neuron_type,name='D5',color='green')
net.addNeuron(neuron_type,name='D10',color='red')
net.addNeuron(neuron_type,name='D20',color='purple')

net.addSynapse(synapse_type_d0,'Source','D0')
net.addSynapse(synapse_type_d5,'Source','D5')
net.addSynapse(synapse_type_d10,'Source','D10')
net.addSynapse(synapse_type_d20,'Source','D20')

net.addOutput('Source',name='OSS',spiking=True)
net.addOutput('D0',name='O0S',spiking=True)
net.addOutput('D5',name='O5S',spiking=True)
net.addOutput('D10',name='O10S',spiking=True)
net.addOutput('D20',name='O20S',spiking=True)

net.addInput('Source')

net.renderGraph(view=True)

"""Simulate the network"""
dt = 0.01
tMax = 10
t = np.arange(0,tMax,dt)
inputs = np.zeros([len(t),net.getNumInputs()])          # getNumInputs() gets the number of input nodes in a network
inputs[0:100] = 20.0
data = np.zeros([len(t),net.getNumOutputsActual()])    # getNumOutputsActual gets the number of accessible output
                                                            # nodes in a network (since this net has populations, each
                                                            # population has n output nodes)
# Compile to numpy
model = SNS_Numpy(net,dt=dt)

# Run for all steps
for i in range(len(t)):
    data[i,:] = model.forward(inputs[i,:])

"""Plotting the results"""
plt.figure()
plt.subplot(2,1,1)
plt.plot(t,inputs,color='black')
plt.title('Input Stimulus')
plt.ylabel('Current (nA)')
plt.subplot(2,1,2)
spike_raster_plot(t,data.transpose()[:][:],colors=['blue','orange','green','red','purple'])
plt.title('Network Spike Times')
plt.ylabel('Neuron')
plt.xlabel('t (ms)')


plt.show()