"""
Let's walk through how to use the SNS Toolbox in a live demo!
Will Nourse
January 24th, 2022
"""

from sns_toolbox.design.neurons import NonSpikingNeuron
from sns_toolbox.design.networks import Network
from sns_toolbox.design.connections import NonSpikingSynapse

# Define a neuron type
neuron_type = NonSpikingNeuron(name='Neuron',color='white',
                               membrane_capacitance=5.0,    #nF
                               membrane_conductance=1.0,    #uS
                               bias=0.0)

synapse_excitatory = NonSpikingSynapse()    # default parameters lead to strong excitation
synapse_inhibitory = NonSpikingSynapse(relative_reversal_potential=-40.0) # make an inhibitory synapse

net = Network(name='Tutorial')

net.add_neuron(neuron_type,name='Neuron 1',color='blue')
net.add_neuron(neuron_type,name='Neuron 2')

net.add_connection(synapse_excitatory,'Neuron 1','Neuron 2')
net.add_connection(synapse_inhibitory,'Neuron 2','Neuron 1')

net.render_graph(view=True)
