"""
Create a network with excitatory, inhibitory, and modulatory connections. Add input sources and output monitors, and
simulate using numpy.
William Nourse
December 1st, 2021
"""

"""Imports"""
# Import packages and modules for designing the network
from sns_toolbox.neurons import NonSpikingNeuron
from sns_toolbox.connections import NonSpikingSynapse
from sns_toolbox.networks import Network
from sns_toolbox.renderer import render

# Import packages and modules for simulating the network
import numpy as np
import matplotlib.pyplot as plt

"""Design the network"""
# Define a non-spiking neuron and excitatory/inhibitory connections as in tutorial_1
neuron_type = NonSpikingNeuron()
synapse_excitatory = NonSpikingSynapse(max_conductance=1.0, reversal_potential=50.0)
synapse_inhibitory = NonSpikingSynapse(max_conductance=1.0, reversal_potential=-40.0)

# Define a modulatory synapse (relative reversal potential is 0)
synapse_modulatory = NonSpikingSynapse(max_conductance=1.0, reversal_potential=0.0)

# Create a network
net = Network(name='Tutorial 2 Network')

# Make a neuron to receive external input
net.add_neuron(neuron_type, name='SourceNrn', color='black')

# Make a neuron which is excited by the source neuron
net.add_neuron(neuron_type, name='Dest1', color='blue')
net.add_connection(synapse_excitatory, 'SourceNrn', 'Dest1')

# Make 2 neurons. One is excited by the source and excites the other, the other inhibits the first
net.add_neuron(neuron_type, name='Dest2', color='orange')
net.add_neuron(neuron_type, name='Dest2In', color='green')
net.add_connection(synapse_excitatory, 'SourceNrn', 'Dest2')
net.add_connection(synapse_excitatory, 'Dest2', 'Dest2In')
net.add_connection(synapse_inhibitory, 'Dest2In', 'Dest2')

# Make one more neuron. This neuron will be both excited by the source neuron and modulated by Dest1
net.add_neuron(neuron_type, name='Dest3', color='red')
net.add_connection(synapse_excitatory, 'SourceNrn', 'Dest3')
net.add_connection(synapse_modulatory, 'Dest1', 'Dest3')

# Add an input source to apply stimulus to the Source Neuron
net.add_input(dest='SourceNrn', name='Input', color='white')
# net.addInputConnection(1.0,'Input','SourceNrn')

# Add output monitors so we can view the state of each neuron in the network
net.add_output('SourceNrn', name='OutSourceNrn', color='grey')
net.add_output('Dest1', name='OutDest1', color='grey')
net.add_output('Dest2', name='OutDest2', color='grey')
net.add_output('Dest2In', name='OutDest2In', color='grey')
net.add_output('Dest3', name='OutDest3', color='grey')

# View the graph of our network, to make sure everything is as we designed it
render(net,view=True)
"""Prep the Simulation"""
# Set simulation parameters
dt = 0.01
t_max = 50

# Initialize a vector of timesteps
t = np.arange(0, t_max, dt)

# Initialize vectors which store the input to our network, and for data to be written to during simulation from outputs
inputs = np.zeros([len(t),1])+20.0  # Input vector must be 2d, even if second dimension is 1
data = np.zeros([len(t),5])

# Compile the network to use the Numpy CPU backend (if you want to see what's happening, set debug to true)
model = net.compile(dt=dt,backend='numpy',debug=False)

"""Simulate the network"""
# At every step, apply the current input to a forward pass of the network and store the results in 'data'
for i in range(len(t)):
    data[i, :] = model(inputs[i, :])
data = data.transpose()

"""Plot the data"""
# First section
plt.figure()
plt.title('First Section')
plt.plot(t,data[:][0],label='SourceNrn',color='black')  # When plotting, all data needs to be transposed first
plt.plot(t,data[:][1],label='Dest1',color='blue')
plt.legend()

# Second section
plt.figure()
plt.title('Second Section')
plt.plot(t,data[:][0],label='SourceNrn',color='black')
plt.plot(t,data[:][2],label='Dest2',color='orange')
plt.plot(t,data[:][3],label='Dest2In',color='green')
plt.legend()

# Third section
plt.figure()
plt.title('Third Section')
plt.plot(t,data[:][0],label='SourceNrn',color='black')
plt.plot(t,data[:][1],label='Dest1',color='blue')
plt.plot(t,data[:][4],label='Dest3',color='red')
plt.legend()

plt.show()  # Show the plots
