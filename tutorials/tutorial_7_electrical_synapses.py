"""
Tutorial demonstrating a network with electrical synapses, both bidirectional and rectified.
"""

from sns_toolbox.design.connections import ElectricalSynapse, NonSpikingTransmissionSynapse
from sns_toolbox.design.neurons import NonSpikingNeuron
from sns_toolbox.design.networks import Network

from sns_toolbox.simulate.backends import SNS_Numpy

import numpy as np
import matplotlib.pyplot as plt


neuron_type = NonSpikingNeuron()
chem = NonSpikingTransmissionSynapse(gain=1)
electric = ElectricalSynapse(1)
electric_rectified = ElectricalSynapse(1,rect=True)

net = Network('Tutorial 7 Network')
net.add_neuron(neuron_type,name='0',color='blue')
net.add_neuron(neuron_type,name='1',color='orange')
net.add_connection(chem,'0','1')
net.add_input('0')
net.add_output('0')
net.add_output('1')

net.add_neuron(neuron_type,name='2',color='green')
net.add_neuron(neuron_type,name='3',color='red')
net.add_connection(electric,'2','3')
net.add_input('2')
net.add_output('2')
net.add_output('3')

net.add_neuron(neuron_type,name='4',color='purple')
net.add_neuron(neuron_type,name='5',color='brown')
net.add_connection(electric_rectified,'4','5')
net.add_input('4')
net.add_output('4')
net.add_output('5')

net.add_neuron(neuron_type,name='6',color='pink')
net.add_neuron(neuron_type,name='7',color='grey')
net.add_connection(electric_rectified,'6','7')
net.add_input('7')
net.add_output('6')
net.add_output('7')

net.render_graph(view=True)

"""Prep the Simulation"""
# Set simulation parameters
dt = 0.01
t_max = 50

# Initialize a vector of timesteps
t = np.arange(0, t_max, dt)

# Initialize vectors which store the input to our network, and for data to be written to during simulation from outputs
inputs = np.zeros([len(t),4])+20.0  # Input vector must be 2d, even if second dimension is 1
data = np.zeros([len(t),net.get_num_outputs_actual()])

# Compile the network to use the Numpy CPU backend (if you want to see what's happening, set debug to true)

model = SNS_Numpy(net, dt=dt, debug=True)

"""Simulate the network"""
# At every step, apply the current input to a forward pass of the network and store the results in 'data'
for i in range(len(t)):
    data[i,:] = model.forward(inputs[i,:])
data = data.transpose()

"""Plot the data"""
plt.figure()
plt.subplot(2,2,1)
plt.plot(t,data[:][0],label='0',color='C0')
plt.plot(t,data[:][1],label='1',color='C1')
plt.xlabel('t (ms)')
plt.ylabel('U (mV)')
plt.title('Non-spiking Chemical Synapse')
plt.legend()

plt.subplot(2,2,2)
plt.plot(t,data[:][2],label='2',color='C2')
plt.plot(t,data[:][3],label='3',color='C3')
plt.xlabel('t (ms)')
plt.ylabel('U (mV)')
plt.title('Electrical Synapse')
plt.legend()

plt.subplot(2,2,3)
plt.plot(t,data[:][4],label='4',color='C4')
plt.plot(t,data[:][5],label='5',color='C5')
plt.xlabel('t (ms)')
plt.ylabel('U (mV)')
plt.title('Rectified Electrical Synapse (Forward)')
plt.legend()

plt.subplot(2,2,4)
plt.plot(t,data[:][6],label='6',color='C6')
plt.plot(t,data[:][7],label='7',color='C7')
plt.xlabel('t (ms)')
plt.ylabel('U (mV)')
plt.title('Rectified Electrical Synapse (Backward)')
plt.legend()

plt.show()
