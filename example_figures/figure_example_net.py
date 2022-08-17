"""
Example network for the living machines paper
William Nourse
January 27th 2022
"""
import numpy as np
import matplotlib.pyplot as plt

from sns_toolbox.neurons import NonSpikingNeuron
from sns_toolbox.connections import NonSpikingSynapse
from sns_toolbox.networks import Network
from sns_toolbox.renderer import render


neuron_type = NonSpikingNeuron()
slow_neuron_type = NonSpikingNeuron(membrane_capacitance=50.0)
synapse_excitatory = NonSpikingSynapse(reversal_potential=40.0)
synapse_inhibitory = NonSpikingSynapse(max_conductance=1.0, reversal_potential=-40.0)
synapse_modulatory = NonSpikingSynapse(reversal_potential=0.0)

net = Network(name='Network')
net.add_neuron(neuron_type,name='0',color='cornflowerblue')
net.add_neuron(neuron_type,name='1',color='darkorange')
net.add_neuron(slow_neuron_type,name='2',color='firebrick')

net.add_connection(synapse_excitatory,'0','1')
net.add_connection(synapse_excitatory,'0','2')
net.add_connection(synapse_modulatory,'1','1')
net.add_connection(synapse_inhibitory,'2','0')

net.add_input('0',name='Iapp')
net.add_output('0',name='O0')
net.add_output('1',name='O1')
net.add_output('2',name='O2')

render(net,view=True, save=True, filename='DocsExample', img_format='png')
# net.render_graph(view=True,imgFormat='svg')

# Set simulation parameters
# dt = 0.01
# t_max = 50
#
# # Initialize a vector of timesteps
# t = np.arange(0, t_max, dt)
#
# # Initialize vectors which store the input to our network, and for data to be written to during simulation from outputs
# inputs = np.zeros([len(t),1])+20.0  # Input vector must be 2d, even if second dimension is 1
# data = np.zeros([len(t),3])
#
# # Compile the network to use the Numpy CPU backend (if you want to see what's happening, set debug to true)
# model = SNS_Numpy(net, delay=delay, spiking=spiking, dt=dt, debug=False)
#
# """Simulate the network"""
# # At every step, apply the current input to a forward pass of the network and store the results in 'data'
# for i in range(len(t)):
#     data[i,:] = model.forward(inputs[i,:])
#
# """Plot the data"""
# # First section
# plt.figure()
# # plt.title('First Section')
# plt.plot(t,data.transpose()[:][0],label='SourceNrn',color='blue')  # When plotting, all data needs to be transposed first
# plt.plot(t,data.transpose()[:][1],label='SourceNrn',color='orange',linestyle='dashed')
# plt.plot(t,data.transpose()[:][2],label='SourceNrn',color='red',linestyle='dotted')
# # plt.legend()
#
# # # Second section
# # plt.figure()
# # # plt.title('Second Section')
# # plt.plot(t,data.transpose()[:][1],label='SourceNrn',color='orange')
# # # plt.legend()
# #
# # # Third section
# # plt.figure()
# # # plt.title('Third Section')
# # plt.plot(t,data.transpose()[:][0],label='SourceNrn',color='red')
# # # plt.legend()
#
# plt.show()  # Show the plots