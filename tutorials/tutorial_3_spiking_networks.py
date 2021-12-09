"""
Create a network of spiking neurons and populations, and record activity with spike monitors
William Nourse
December 2nd 2021
"""

"""Imports"""
# Import packages and modules for designing the network
from sns_toolbox.design.networks import Network
from sns_toolbox.design.connections import SpikingSynapse
from sns_toolbox.design.neurons import SpikingNeuron

# Import packages and modules for simulating the network
from sns_toolbox.simulate.backends import SNS_Numpy
import numpy as np
import matplotlib.pyplot as plt
from sns_toolbox.simulate.plotting import spike_raster_plot

"""Design the first Network"""
# Create spiking neurons with different values of 'm'
spike_m_equal_0 = SpikingNeuron(name='m = 0', color='aqua',
                                thresholdTimeConstant=5.0,   # Default value of tau_m (ms)
                                thresholdProportionalityConstant=0.0,    # Default value of m
                                thresholdInitialValue=1.0)  # Default value of theta_0 (mV)
spike_m_less_0 = SpikingNeuron(name='m < 0', color='darkorange',
                               thresholdProportionalityConstant=-1.0)
spike_m_greater_0 = SpikingNeuron(name='m > 0', color='forestgreen',
                                  thresholdProportionalityConstant=1.0)

# Create a spiking synapse
synapse_spike = SpikingSynapse(synapticTimeConstant=1.0)    # Default value (ms)

# Create a network with different m values
net = Network(name='Tutorial 3 Network Neurons')
net.addNeuron(spike_m_equal_0, name='m=0')
net.addNeuron(spike_m_less_0, name='m<0')
net.addNeuron(spike_m_greater_0, name='m>0')

# Add an input current source
net.addInput(dest='m=0',name='I0', color='black')
net.addInput(dest='m<0',name='I1', color='black')
net.addInput(dest='m>0',name='I2', color='black')
# net.addInputConnection(1, 'I0', 'm=0')
# net.addInputConnection(1, 'I0', 'm<0')
# net.addInputConnection(1, 'I0', 'm>0')

# Add output monitors (some for the voltage, some for the spikes)
net.addOutput('m=0', name='O0V', color='grey')
net.addOutput('m=0', name='O1S', color='grey', spiking=True)  # Records spikes instead of voltage
net.addOutput('m<0', name='O2V', color='grey')
net.addOutput('m<0', name='O3S', color='grey', spiking=True)  # Records spikes instead of voltage
net.addOutput('m>0', name='O4V', color='grey')
net.addOutput('m>0', name='O5S', color='grey', spiking=True)  # Records spikes instead of voltage

net.renderGraph(view=False)

"""Define the second network"""
pop_size = 5
net_pop = Network(name='Tutorial 3 Network Populations')
net_pop.addPopulation(spike_m_equal_0,numNeurons=pop_size,color='red',name='Source')
net_pop.addPopulation(spike_m_equal_0,numNeurons=pop_size,color='purple',name='Destination')
net_pop.addInput(dest='Source',name='I3',color='black')
net_pop.addSynapse(synapse_spike,'Source','Destination')
net_pop.addOutput('Source',name='O6S',color='grey',spiking=True)
net_pop.addOutput('Source',name='O7V',color='grey',spiking=False)
net_pop.addOutput('Destination',name='O8S', color='grey', spiking=True)
net_pop.addOutput('Destination',name='O9V',color='grey',spiking=False)

net_pop.renderGraph(view=False)

"""Combine both networks into one for easier simulation"""
net_comb = Network(name='Tutorial 3 Network Combined')
net_comb.addNetwork(net)
net_comb.addNetwork(net_pop)

net_comb.renderGraph(view=True)

"""Simulate both networks"""
dt = 0.01
tMax = 10
t = np.arange(0,tMax,dt)
inputs = np.zeros([len(t),net_comb.getNumInputs()])+20      # getNumInputs() gets the number of input nodes in a network
data = np.zeros([len(t),net_comb.getNumOutputsActual()])    # getNumOutputsActual gets the number of accessible output
                                                            # nodes in a network (since this net has populations, each
                                                            # population has n output nodes)
# Compile to numpy
model = SNS_Numpy(net_comb,dt=dt)

# Run for all steps
for i in range(len(t)):
    data[i,:] = model.forward(inputs[i,:])

"""Plotting the results"""
# First network
plt.figure()
plt.subplot(3,2,1)
plt.title('m = 0: Voltage')
plt.plot(t,data.transpose()[:][0],color='blue')
# plt.xlabel('t (ms)')
plt.ylabel('U (mV)')
plt.subplot(3,2,2)
plt.title('m = 0: Spikes')
spike_raster_plot(t, data.transpose()[:][1],colors=['blue'])
# plt.xlabel('t (ms)')
plt.subplot(3,2,3)
plt.title('m < 0: Voltage')
plt.plot(t,data.transpose()[:][2],color='orange')
# plt.xlabel('t (ms)')
plt.ylabel('U (mV)')
plt.subplot(3,2,4)
plt.title('m = 0: Spikes')
spike_raster_plot(t, data.transpose()[:][3],colors=['orange'])
# plt.xlabel('t (ms)')
plt.subplot(3,2,5)
plt.title('m > 0: Voltage')
plt.plot(t,data.transpose()[:][4],color='green')
plt.xlabel('t (ms)')
plt.ylabel('U (mV)')
plt.subplot(3,2,6)
plt.title('m > 0: Spikes')
spike_raster_plot(t, data.transpose()[:][5],colors=['green'])
plt.xlabel('t (ms)')

plt.figure()
plt.subplot(2,2,1)
spike_raster_plot(t,data.transpose()[:][6:6+pop_size],colors=['red'])
plt.ylabel('Neuron')
plt.title('Source Spikes')
plt.subplot(2, 2, 2)
spike_raster_plot(t,data.transpose()[:][6+pop_size:],colors=['purple'])
plt.ylabel('Neuron')
plt.title('Destination Spikes')
plt.subplot(2,2,3)
for i in range(pop_size):
    plt.plot(t,data.transpose()[:][6+i])
plt.xlabel('t (ms)')
plt.ylabel('Voltage')
plt.title('Source Voltage')
plt.subplot(2, 2, 4)
for i in range(pop_size):
    plt.plot(t,data.transpose()[:][6+pop_size+i])
plt.xlabel('t (ms)')
plt.ylabel('Voltage')
plt.title('Destination Voltage')

plt.show()