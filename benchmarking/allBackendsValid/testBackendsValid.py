"""
Make networks with all of the components, and make sure they all compile and perform the same.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch

from sns_toolbox.design.networks import Network
from sns_toolbox.design.neurons import NonSpikingNeuron, SpikingNeuron
from sns_toolbox.design.connections import NonSpikingSynapse, SpikingSynapse

import sns_toolbox.simulate.backends
from sns_toolbox.simulate.simulate_utilities import spike_raster_plot

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Network 1: NonSpiking
"""
# No spiking, no delay
spiking = False
delay = False


neuron_type = NonSpikingNeuron()
synapse_excitatory = NonSpikingSynapse(max_conductance=1.0, relative_reversal_potential=50.0)
synapse_inhibitory = NonSpikingSynapse(max_conductance=1.0, relative_reversal_potential=-40.0)
synapse_modulatory = NonSpikingSynapse(max_conductance=1.0, relative_reversal_potential=0.0)
net = Network(name='Tutorial 2 Network',R=20.0)
net.add_neuron(neuron_type, name='SourceNrn', color='black')
net.add_neuron(neuron_type, name='Dest1', color='blue')
net.add_connection(synapse_excitatory, 'SourceNrn', 'Dest1')
net.add_neuron(neuron_type, name='Dest2', color='orange')
net.add_neuron(neuron_type, name='Dest2In', color='green')
net.add_connection(synapse_excitatory, 'SourceNrn', 'Dest2')
net.add_connection(synapse_excitatory, 'Dest2', 'Dest2In')
net.add_connection(synapse_inhibitory, 'Dest2In', 'Dest2')
net.add_neuron(neuron_type, name='Dest3', color='red')
net.add_connection(synapse_excitatory, 'SourceNrn', 'Dest3')
net.add_connection(synapse_modulatory, 'Dest1', 'Dest3')
net.add_input(dest='SourceNrn', name='Input', color='white')
net.add_output('SourceNrn', name='OutSourceNrn', color='grey')
net.add_output('Dest1', name='OutDest1', color='grey')
net.add_output('Dest2', name='OutDest2', color='grey')
net.add_output('Dest2In', name='OutDest2In', color='grey')
net.add_output('Dest3', name='OutDest3', color='grey')

"""Prep the Simulation"""
# Set simulation parameters
dt = 0.01
t_max = 50

# Initialize a vector of timesteps
t = np.arange(0, t_max, dt)

# Initialize vectors which store the input to our network, and for data to be written to during simulation from outputs
inputs = np.zeros([len(t),1])+20.0  # Input vector must be 2d, even if second dimension is 1
inputsTorch = torch.zeros([len(t),1])+20.0  # Input vector must be 2d, even if second dimension is 1
dataNumpy = np.zeros([len(t),5])
dataTorch = torch.zeros([len(t),5])
dataSparse = torch.zeros([len(t),5])
dataManual = np.zeros([len(t),5])

# Compile the network to use the Numpy CPU backend (if you want to see what's happening, set debug to true)

modelNumpy = sns_toolbox.simulate.backends.SNS_Numpy(net, dt=dt, debug=False,spiking=spiking, delay=delay)
modelTorch = sns_toolbox.simulate.backends.SNS_Torch(net, dt=dt, debug=False,spiking=spiking, delay=delay, device='cpu')
modelSparse = sns_toolbox.simulate.backends.SNS_Sparse(net, dt=dt, debug=False,spiking=spiking, delay=delay, device='cpu')
modelManual = sns_toolbox.simulate.backends.SNS_Manual(net, dt=dt, debug=False,spiking=spiking, delay=delay)

"""Simulate the network"""
print('Running Network 1')
for i in range(len(t)):
    print('1: %i / %i steps' % (i + 1, len(t)))
    dataNumpy[i,:] = modelNumpy.forward(inputs[i,:])
    dataTorch[i, :] = modelTorch.forward(inputsTorch[i, :])
    dataSparse[i, :] = modelSparse.forward(inputsTorch[i, :])
    dataManual[i, :] = modelManual.forward(inputs[i, :])
dataNumpy = dataNumpy.transpose()
dataTorch = torch.transpose(dataTorch,0,1)
dataSparse = torch.transpose(dataSparse,0,1)
dataManual = dataManual.transpose()

"""Plot the data"""
plt.figure()
plt.title('NonSpiking Network')
plt.plot(t,dataNumpy[:][0],color='black')  # When plotting, all data needs to be transposed first
plt.plot(t,dataNumpy[:][1],color='blue')
plt.plot(t,dataNumpy[:][2],color='orange')
plt.plot(t,dataNumpy[:][3],color='green')
plt.plot(t,dataNumpy[:][4],color='red')
plt.plot(t,dataTorch[:][0],color='black',linestyle='dotted')
plt.plot(t,dataTorch[:][1],color='blue',linestyle='dotted')
plt.plot(t,dataTorch[:][2],color='orange',linestyle='dotted')
plt.plot(t,dataTorch[:][3],color='green',linestyle='dotted')
plt.plot(t,dataTorch[:][4],color='red',linestyle='dotted')
plt.plot(t,dataSparse[:][0],color='black',linestyle='dashed')
plt.plot(t,dataSparse[:][1],color='blue',linestyle='dashed')
plt.plot(t,dataSparse[:][2],color='orange',linestyle='dashed')
plt.plot(t,dataSparse[:][3],color='green',linestyle='dashed')
plt.plot(t,dataSparse[:][4],color='red',linestyle='dashed')
plt.plot(t,dataManual[:][0],color='black',linestyle='dashdot')
plt.plot(t,dataManual[:][1],color='blue',linestyle='dashdot')
plt.plot(t,dataManual[:][2],color='orange',linestyle='dashdot')
plt.plot(t,dataManual[:][3],color='green',linestyle='dashdot')
plt.plot(t,dataManual[:][4],color='red',linestyle='dashdot')

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Network 2: Spiking
"""
spiking = True
delay = False
threshold_initial_value = 1.0
spike_m_equal_0 = SpikingNeuron(name='m = 0', color='aqua',
                                threshold_time_constant=5.0,  # Default value of tau_m (ms)
                                threshold_proportionality_constant=0.0,  # Default value of m
                                threshold_initial_value=threshold_initial_value)  # Default value of theta_0 (mV)
spike_m_less_0 = SpikingNeuron(name='m < 0', color='darkorange',
                               threshold_proportionality_constant=-1.0)
spike_m_greater_0 = SpikingNeuron(name='m > 0', color='forestgreen',
                                  threshold_proportionality_constant=1.0)
synapse_spike = SpikingSynapse(time_constant=1.0)    # Default value (ms)
net = Network(name='Tutorial 3 Network Neurons')
net.add_neuron(spike_m_equal_0, name='m=0')
net.add_neuron(spike_m_less_0, name='m<0')
net.add_neuron(spike_m_greater_0, name='m>0')
net.add_input(dest='m=0', name='I0', color='black')
net.add_input(dest='m<0', name='I1', color='black')
net.add_input(dest='m>0', name='I2', color='black')
net.add_output('m=0', name='O0V', color='grey')
net.add_output('m<0', name='O2V', color='grey')
net.add_output('m>0', name='O4V', color='grey')

"""Simulate both networks"""
dt = 0.01
t_max = 10
t = np.arange(0, t_max, dt)
inputs = np.zeros([len(t), net.get_num_inputs()]) + 20      # getNumInputs() gets the number of input nodes in a network
inputsTorch = torch.zeros([len(t), net.get_num_inputs()]) + 20
dataNumpy = np.zeros([len(t), net.get_num_outputs_actual()])
dataTorch = torch.zeros([len(t), net.get_num_outputs_actual()])
dataSparse = torch.zeros([len(t), net.get_num_outputs_actual()])
dataManual = np.zeros([len(t), net.get_num_outputs_actual()])

modelNumpy = sns_toolbox.simulate.backends.SNS_Numpy(net, dt=dt, debug=False,spiking=spiking, delay=delay)
modelTorch = sns_toolbox.simulate.backends.SNS_Torch(net, dt=dt, debug=False,spiking=spiking, delay=delay, device='cpu')
modelSparse = sns_toolbox.simulate.backends.SNS_Sparse(net, dt=dt, debug=False,spiking=spiking, delay=delay, device='cpu')
modelManual = sns_toolbox.simulate.backends.SNS_Manual(net, dt=dt, debug=False,spiking=spiking, delay=delay)

"""Simulate the network"""
print('Running Network 2')
for i in range(len(t)):
    print('2: %i / %i steps' % (i + 1, len(t)))
    dataNumpy[i,:] = modelNumpy.forward(inputs[i,:])
    dataTorch[i, :] = modelTorch.forward(inputsTorch[i, :])
    dataSparse[i, :] = modelSparse.forward(inputsTorch[i, :])
    dataManual[i, :] = modelManual.forward(inputs[i, :])
dataNumpy = dataNumpy.transpose()
dataTorch = torch.transpose(dataTorch,0,1)
dataSparse = torch.transpose(dataSparse,0,1)
dataManual = dataManual.transpose()

plt.figure()
plt.subplot(3,1,1)
plt.title('m = 0: Voltage')
plt.plot(t,dataNumpy[:][0],color='blue')
plt.plot(t,dataTorch[:][0],color='blue',linestyle='dotted')
plt.plot(t,dataSparse[:][0],color='blue',linestyle='dashed')
plt.plot(t,dataManual[:][0],color='blue',linestyle='dashdot')
plt.ylabel('u (mV)')
plt.subplot(3,1,2)
plt.title('m < 0: Voltage')
plt.plot(t,dataNumpy[:][1],color='orange')
plt.plot(t,dataTorch[:][1],color='orange',linestyle='dotted')
plt.plot(t,dataSparse[:][1],color='orange',linestyle='dashed')
plt.plot(t,dataManual[:][1],color='orange',linestyle='dashdot')
plt.ylabel('u (mV)')
plt.subplot(3,1,3)
plt.title('m > 0: Voltage')
plt.plot(t,dataNumpy[:][2],color='green')
plt.plot(t,dataTorch[:][2],color='green',linestyle='dotted')
plt.plot(t,dataSparse[:][2],color='green',linestyle='dashed')
plt.plot(t,dataManual[:][2],color='green',linestyle='dashdot')
plt.xlabel('t (ms)')
plt.ylabel('u (mV)')

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Network 3: Transmission Delay
"""
spiking = True
delay = True

neuron_type = SpikingNeuron()
synapse_type_d20 = SpikingSynapse(transmission_delay=20)   # Transmission delay of 20 dt
net = Network(name='Tutorial 5 Network')
net.add_neuron(neuron_type, name='Source', color='blue')
net.add_neuron(neuron_type, name='D20', color='purple')
net.add_connection(synapse_type_d20, 'Source', 'D20')
net.add_output('D20', name='O20S', spiking=True)
net.add_input('Source')

dt = 0.01
t_max = 10
t = np.arange(0, t_max, dt)
inputs = np.zeros([len(t), net.get_num_inputs()]) + 20.0
inputsTorch = torch.zeros([len(t), net.get_num_inputs()]) + 20.0
dataNumpy = np.zeros([len(t), net.get_num_outputs_actual()])
dataTorch = torch.zeros([len(t), net.get_num_outputs_actual()])
dataSparse = torch.zeros([len(t), net.get_num_outputs_actual()])
dataManual = np.zeros([len(t), net.get_num_outputs_actual()])

modelNumpy = sns_toolbox.simulate.backends.SNS_Numpy(net, dt=dt, debug=False,spiking=spiking, delay=delay)
modelTorch = sns_toolbox.simulate.backends.SNS_Torch(net, dt=dt, debug=False,spiking=spiking, delay=delay, device='cpu')
modelSparse = sns_toolbox.simulate.backends.SNS_Sparse(net, dt=dt, debug=False,spiking=spiking, delay=delay, device='cpu')
modelManual = sns_toolbox.simulate.backends.SNS_Manual(net, dt=dt, debug=False,spiking=spiking, delay=delay)

print('Running Network 3')
for i in range(len(t)):
    print('3: %i / %i steps' % (i + 1, len(t)))
    dataNumpy[i,:] = modelNumpy.forward(inputs[i,:])
    dataTorch[i, :] = modelTorch.forward(inputsTorch[i, :])
    dataSparse[i, :] = modelSparse.forward(inputsTorch[i, :])
    dataManual[i, :] = modelManual.forward(inputs[i, :])
dataNumpy = dataNumpy.transpose()
dataTorch = torch.transpose(dataTorch,0,1)
dataSparse = torch.transpose(dataSparse,0,1)
dataManual = dataManual.transpose()

"""Plotting the results"""
plt.figure()
spike_raster_plot(t,dataNumpy[:][:],colors=['blue'])
spike_raster_plot(t,dataTorch[:][:],colors=['orange'],offset=1)
spike_raster_plot(t,dataSparse[:][:],colors=['green'],offset=2)
spike_raster_plot(t,dataManual[:][:],colors=['red'],offset=3)
plt.title('Network Spike Times')
plt.ylabel('Neuron')
plt.xlabel('t (ms)')

plt.show()