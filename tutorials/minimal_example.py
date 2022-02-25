from sns_toolbox.design.neurons import NonSpikingNeuron
from sns_toolbox.design.connections import NonSpikingSynapse
from sns_toolbox.design.networks import Network
import numpy as np
from sns_toolbox.simulate.backends import SNS_Numpy

# Design
neuron_type = NonSpikingNeuron()    # Define a neuron type
synapse = NonSpikingSynapse()   # Define a non-spiking synapse
net = Network(name='Minimal Network Example') # Create a network
# Add two neurons to the network
net.add_neuron(neuron_type, name='SourceNrn')
net.add_neuron(neuron_type, name='DestNrn')
# Add a synaptic connections between these neurons
net.add_connection(synapse, 'SourceNrn', 'DestNrn')
# Add an external current source
net.add_input(dest='SourceNrn', name='Input')
# Add output monitors to view the state of each neuron
net.add_output('SourceNrn', name='OutSourceNrn')
net.add_output('DestNrn', name='OutDestNrn')
# View the network graph
net.render_graph(view=True)

# Simulation
dt = 0.01   # simulation timestep
num_steps = 50  # number of simulation steps to execute
stim_mag = 20.0 # nA
inputs = np.zeros([num_steps,1])+stim_mag  # Input vector
# Compile the network to use the Numpy-based CPU backend
model = SNS_Numpy(net,dt=dt)
# At every step, apply the current input to a forward pass of the network and get the vector of output monitor states
for i in range(num_steps):
    outputs = model.forward(inputs[i,:])
