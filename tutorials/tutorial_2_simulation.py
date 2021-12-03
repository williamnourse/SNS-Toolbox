"""
Create a network with excitatory, inhibitory, and modulatory synapses. Add input sources and output monitors, and
simulate using numpy.
William Nourse
December 1st, 2021
"""

"""Imports"""
# Import packages and modules for designing the network
from sns_toolbox.design.neurons import NonSpikingNeuron
from sns_toolbox.design.connections import NonSpikingSynapse
from sns_toolbox.design.networks import Network

# Import packages and modules for simulating the network
import numpy as np
import matplotlib.pyplot as plt
from sns_toolbox.simulate.backends import SNS_Numpy

"""Design the network"""
# Define a non-spiking neuron and excitatory/inhibitory synapses as in tutorial_1
neuron_type = NonSpikingNeuron()
synapse_excitatory = NonSpikingSynapse(maxConductance=1.0,relativeReversalPotential=50.0)
synapse_inhibitory = NonSpikingSynapse(maxConductance=1.0,relativeReversalPotential=-40.0)

# Define a modulatory synapse (relative reversal potential is 0)
synapse_modulatory = NonSpikingSynapse(maxConductance=1.0,relativeReversalPotential=0.0)

# Create a network
net = Network(name='Tutorial 2 Network',R=20.0)

# Make a neuron to receive external input
net.addNeuron(neuron_type,name='SourceNrn',color='black')

# Make a neuron which is excited by the source neuron
net.addNeuron(neuron_type,name='Dest1',color='blue')
net.addSynapse(synapse_excitatory,'SourceNrn','Dest1')

# Make 2 neurons. One is excited by the source and excites the other, the other inhibits the first
net.addNeuron(neuron_type,name='Dest2',color='orange')
net.addNeuron(neuron_type,name='Dest2In',color='green')
net.addSynapse(synapse_excitatory,'SourceNrn','Dest2')
net.addSynapse(synapse_excitatory,'Dest2','Dest2In')
net.addSynapse(synapse_inhibitory,'Dest2In','Dest2')

# Make one more neuron. This neuron will be both excited by the source neuron and modulated by Dest1
net.addNeuron(neuron_type,name='Dest3',color='red')
net.addSynapse(synapse_excitatory,'SourceNrn','Dest3')
net.addSynapse(synapse_modulatory,'Dest1','Dest3')

# Add an input source to apply stimulus to the Source Neuron
net.addInput(name='Input',color='white')
net.addInputConnection(1.0,'Input','SourceNrn')

# Add output monitors so we can view the state of each neuron in the network
net.addOutput('SourceNrn',name='OutSourceNrn',color='grey')
net.addOutput('Dest1',name='OutDest1',color='grey')
net.addOutput('Dest2',name='OutDest2',color='grey')
net.addOutput('Dest2In',name='OutDest2In',color='grey')
net.addOutput('Dest3',name='OutDest3',color='grey')

# View the graph of our network, to make sure everything is as we designed it
net.renderGraph(view=True)

"""Prep the Simulation"""
# Set simulation parameters
dt = 0.01
tMax = 10

# Initialize a vector of timesteps
t = np.arange(0,tMax,dt)

# Initialize vectors which store the input to our network, and for data to be written to during simulation from outputs
inputs = np.zeros([len(t),1])+20.0  # Input vector must be 2d, even if second dimension is 1
data = np.zeros([len(t),5])

# Compile the network to use the Numpy CPU backend (if you want to see what's happening, set debug to true)
model = SNS_Numpy(net,dt=dt,debug=False)

"""Simulate the network"""
# At every step, apply the current input to a forward pass of the network and store the results in 'data'
for i in range(len(t)):
    data[i,:] = model.forward(inputs[i,:])

"""Plot the data"""
# First section
plt.figure()
plt.title('First Section')
plt.plot(t,data.transpose()[:][0],label='SourceNrn',color='black')  # When plotting, all data needs to be transposed first
plt.plot(t,data.transpose()[:][1],label='Dest1',color='blue')
plt.legend()

# Second section
plt.figure()
plt.title('Second Section')
plt.plot(t,data.transpose()[:][0],label='SourceNrn',color='black')
plt.plot(t,data.transpose()[:][2],label='Dest2',color='orange')
plt.plot(t,data.transpose()[:][3],label='Dest2In',color='green')
plt.legend()

# Third section
plt.figure()
plt.title('Third Section')
plt.plot(t,data.transpose()[:][0],label='SourceNrn',color='black')
plt.plot(t,data.transpose()[:][1],label='Dest1',color='blue')
plt.plot(t,data.transpose()[:][4],label='Dest3',color='red')
plt.legend()

plt.show()  # Show the plots