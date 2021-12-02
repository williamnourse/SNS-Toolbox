"""
Create a network with excitatory, inhibitory, and modulatory synapses. Add input sources and output monitors, and
simulate using numpy.
William Nourse
December 1st, 2021
"""

"""Imports"""
# Import packages and modules for designing the network
import sns_toolbox
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
synapse_excitatory = NonSpikingSynapse(maxConductance=1.0,relativeReversalPotential=100.0)
synapse_inhibitory = NonSpikingSynapse(maxConductance=1.0,relativeReversalPotential=-40.0)

# Define a modulatory synapse (relative reversal potential is 0)
synapse_modulatory = NonSpikingSynapse(maxConductance=0.5,relativeReversalPotential=0.0)

# Create a network
net = Network(name='Tutorial 2 Network',R=20.0)

# Make a neuron to receive external input
net.addNeuron(neuron_type,name='SourceNrn',color='white')

# Make a neuron which is excited by the source neuron
net.addNeuron(neuron_type,name='Dest1',color='blue')
net.addSynapse(synapse_excitatory,'SourceNrn','Dest1')

# Make 2 neurons. One is excited by the source and excites the other, the other inhibits the first
net.addNeuron(neuron_type,name='Dest2',color='orange')
net.addNeuron(neuron_type,name='Dest2In',color='green')
net.addSynapse(synapse_excitatory,'SourceNrn','Dest2')
net.addSynapse(synapse_excitatory,'Dest2','Dest2In')
net.addSynapse(synapse_inhibitory,'Dest2In','Dest2')

# Make one more neuron. This neuron will be both excited and modulated by the source neuron
net.addNeuron(neuron_type,name='Dest3',color='red')
net.addSynapse(synapse_excitatory,'SourceNrn','Dest3')
net.addSynapse(synapse_modulatory,'SourceNrn','Dest3')

# Add an input source to apply stimulus to the Source Neuron
net.addInput(name='Input',color='black')
net.addInputConnection(1.0,'Input','SourceNrn')

# Add output monitors so we can view the state of each neuron in the network
net.addOutput('SourceNrn',name='OutSourceNrn',color='grey')
net.addOutput('Dest1',name='OutDest1',color='grey')
net.addOutput('Dest2',name='OutDest2',color='grey')
net.addOutput('Dest2In',name='OutDest2In',color='grey')
net.addOutput('Dest3',name='OutDest3',color='grey')

# View the graph of our network, to make sure everything is as we designed it
net.renderGraph(view=True)

