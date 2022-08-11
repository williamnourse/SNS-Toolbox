"""
The first in a series of tutorials about how to effectively use the SNS Toolbox. In this tutorial, we design a simple
network with a single neuron and an applied current source.
William Nourse
November 29 2021
"""

"""
1. Import necessary packages and modules/classes
"""
from sns_toolbox.neurons import NonSpikingNeuron
from sns_toolbox.networks import Network
from sns_toolbox.connections import NonSpikingSynapse
from sns_toolbox.renderer import render

"""
2. Define a neuron type. In a network we can have many different types, but for now we're just going to use one
"""
# All of the following parameters are the default values for a non-spiking neuron
neuron_type = NonSpikingNeuron(name='Neuron',  # Name displayed in a render of the network
                               color='white',  # Fill color for the rendered neuron
                               membrane_capacitance=5.0,  # Membrane capacitance in nF
                               membrane_conductance=1.0,  # Membrane conductance in uS
                               bias=0.0)                # Tonic applied current in nA

"""
3. Create a network, and add multiple copies of our neuron to it
"""
# Create the network
net = Network(name='Tutorial 1 Network') # Optional parameters are a name and the value of 'R', in mV

# Add neurons to the network
net.add_neuron(neuron_type, name='Neuron 1', color='blue')
net.add_neuron(neuron_type, name='Neuron 2', color='black')

"""
4. Define some synapse types
"""
synapse_excitatory = NonSpikingSynapse()    # default parameters lead to strong excitation
synapse_inhibitory = NonSpikingSynapse(relative_reversal_potential=-40.0) # make an inhibitory synapse

"""
5. Use our synapse types to connect the neurons in the network
"""
net.add_connection(synapse_excitatory, 'Neuron 1', 'Neuron 2')    # Add an excitatory synapse from neuron 1 to neuron 2
net.add_connection(synapse_inhibitory, 'Neuron 2', 'Neuron 1')   # Add an inhibitory synapse back from 2 to 1

# This can also be done using indices in the order the neurons were added to the network
# net.addSynapse(synapse,0,1)
# net.addSynapse(synapse_inhibitory,1,0)

"""
6. View our network as a visual graph
"""
render(net, view=True)
