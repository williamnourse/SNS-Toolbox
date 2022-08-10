"""
Make networks with all of the components, and make sure they all compile and perform the same.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch

from sns_toolbox.design.networks import Network
from sns_toolbox.design.neurons import NonSpikingNeuron, SpikingNeuron, NonSpikingNeuronWithPersistentSodiumChannel, NonSpikingNeuronWithGatedChannels
from sns_toolbox.design.connections import NonSpikingSynapse, SpikingSynapse, NonSpikingTransmissionSynapse, ElectricalSynapse

from sns_toolbox.simulate.simulate_utilities import spike_raster_plot
import sys

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Network 1: NonSpiking
"""
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

modelNumpy = net.compile(backend='numpy', dt=dt, debug=False)
modelTorch = net.compile(backend='torch', dt=dt, debug=False, device='cpu')
modelSparse = net.compile(backend='sparse', dt=dt, debug=False, device='cpu')
modelManual = net.compile(backend='manual', dt=dt, debug=False)

"""Simulate the network"""
print('Running Network 1')
for i in range(len(t)):
    print('1: %i / %i steps' % (i + 1, len(t)))
    dataNumpy[i,:] = modelNumpy(inputs[i,:])
    dataTorch[i, :] = modelTorch(inputsTorch[i, :])
    dataSparse[i, :] = modelSparse(inputsTorch[i, :])
    dataManual[i, :] = modelManual(inputs[i, :])
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

modelNumpy = net.compile(backend='numpy', dt=dt, debug=False)
modelTorch = net.compile(backend='torch', dt=dt, debug=False, device='cpu')
modelSparse = net.compile(backend='sparse', dt=dt, debug=False, device='cpu')
modelManual = net.compile(backend='manual', dt=dt, debug=False)

"""Simulate the network"""
print('Running Network 2')
for i in range(len(t)):
    print('2: %i / %i steps' % (i + 1, len(t)))
    dataNumpy[i,:] = modelNumpy(inputs[i,:])
    dataTorch[i, :] = modelTorch(inputsTorch[i, :])
    dataSparse[i, :] = modelSparse(inputsTorch[i, :])
    dataManual[i, :] = modelManual(inputs[i, :])
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

modelNumpy = net.compile(backend='numpy', dt=dt, debug=False)
modelTorch = net.compile(backend='torch', dt=dt, debug=False, device='cpu')
modelSparse = net.compile(backend='sparse', dt=dt, debug=False, device='cpu')
modelManual = net.compile(backend='manual', dt=dt, debug=False)

print('Running Network 3')
for i in range(len(t)):
    print('3: %i / %i steps' % (i + 1, len(t)))
    dataNumpy[i,:] = modelNumpy(inputs[i,:])
    dataTorch[i, :] = modelTorch(inputsTorch[i, :])
    dataSparse[i, :] = modelSparse(inputsTorch[i, :])
    dataManual[i, :] = modelManual(inputs[i, :])
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

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Network 4: Electrical Synapses
"""
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

# net.render_graph(view=True)

"""Prep the Simulation"""
# Set simulation parameters
dt = 0.01
t_max = 50

# Initialize a vector of timesteps
t = np.arange(0, t_max, dt)

# Initialize vectors which store the input to our network, and for data to be written to during simulation from outputs
inputsNumpy = np.zeros([len(t),4])+20.0  # Input vector must be 2d, even if second dimension is 1
inputsTorch = torch.zeros([len(t),4])+20.0  # Input vector must be 2d, even if second dimension is 1
dataNumpy = np.zeros([len(t),net.get_num_outputs_actual()])
dataTorch = torch.zeros([len(t),net.get_num_outputs_actual()])
dataManual = np.zeros([len(t),net.get_num_outputs_actual()])
dataSparse = torch.zeros([len(t),net.get_num_outputs_actual()])

# Compile the network to use the Numpy CPU backend (if you want to see what's happening, set debug to true)

modelNumpy = net.compile(backend='numpy', dt=dt, debug=False)
modelTorch = net.compile(backend='torch', dt=dt, debug=False, device='cpu')
modelSparse = net.compile(backend='sparse', dt=dt, debug=False, device='cpu')
modelManual = net.compile(backend='manual', dt=dt, debug=False)

print('Running Network 4')
for i in range(len(t)):
    print('4: %i / %i steps' % (i + 1, len(t)))
    dataNumpy[i,:] = modelNumpy(inputsNumpy[i,:])
    dataTorch[i, :] = modelTorch(inputsTorch[i, :])
    dataSparse[i, :] = modelSparse(inputsTorch[i, :])
    dataManual[i, :] = modelManual(inputsNumpy[i, :])
dataNumpy = dataNumpy.transpose()
dataTorch = torch.transpose(dataTorch,0,1)
dataSparse = torch.transpose(dataSparse,0,1)
dataManual = dataManual.transpose()

"""Plot the data"""
plt.figure()
plt.subplot(2,2,1)
plt.plot(t,dataNumpy[:][0],label='0',color='C0')
plt.plot(t,dataTorch[:][0],label='0',color='C1')
plt.plot(t,dataSparse[:][0],label='0',color='C2')
plt.plot(t,dataManual[:][0],label='0',color='C3')
plt.plot(t,dataNumpy[:][1],label='1',color='C0')
plt.plot(t,dataTorch[:][1],label='1',color='C1')
plt.plot(t,dataSparse[:][1],label='1',color='C2')
plt.plot(t,dataManual[:][1],label='1',color='C3')
plt.xlabel('t (ms)')
plt.ylabel('U (mV)')
plt.title('Non-spiking Chemical Synapse')

plt.subplot(2,2,2)
plt.plot(t,dataNumpy[:][2],label='2',color='C0')
plt.plot(t,dataTorch[:][2],label='2',color='C1')
plt.plot(t,dataSparse[:][2],label='2',color='C2')
plt.plot(t,dataManual[:][2],label='2',color='C3')
plt.plot(t,dataNumpy[:][3],label='3',color='C0')
plt.plot(t,dataTorch[:][3],label='3',color='C1')
plt.plot(t,dataSparse[:][3],label='3',color='C2')
plt.plot(t,dataManual[:][3],label='3',color='C3')
plt.xlabel('t (ms)')
plt.ylabel('U (mV)')
plt.title('Electrical Synapse')

plt.subplot(2,2,3)
plt.plot(t,dataNumpy[:][4],label='2',color='C0')
plt.plot(t,dataTorch[:][4],label='2',color='C1')
plt.plot(t,dataSparse[:][4],label='2',color='C2')
plt.plot(t,dataManual[:][4],label='2',color='C3')
plt.plot(t,dataNumpy[:][5],label='3',color='C0')
plt.plot(t,dataTorch[:][5],label='3',color='C1')
plt.plot(t,dataSparse[:][5],label='3',color='C2')
plt.plot(t,dataManual[:][5],label='3',color='C3')
plt.xlabel('t (ms)')
plt.ylabel('U (mV)')
plt.title('Rectified Electrical Synapse (Forward)')

plt.subplot(2,2,4)
plt.plot(t,dataNumpy[:][6],label='2',color='C0')
plt.plot(t,dataTorch[:][6],label='2',color='C1')
plt.plot(t,dataSparse[:][6],label='2',color='C2')
plt.plot(t,dataManual[:][6],label='2',color='C3')
plt.plot(t,dataNumpy[:][7],label='3',color='C0')
plt.plot(t,dataTorch[:][7],label='3',color='C1')
plt.plot(t,dataSparse[:][7],label='3',color='C2')
plt.plot(t,dataManual[:][7],label='3',color='C3')
plt.xlabel('t (ms)')
plt.ylabel('U (mV)')
plt.title('Rectified Electrical Synapse (Backward)')

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Network 5: Voltage-gated Ion Channels
"""
delta = 0.1
Cm = 5
Gm = 1
Ena = 50
Er = -60
R = 20

S = 0.05
delEna = Ena - Er
Km = 1
Kh = 0.5
Em = -40
Eh = -60
delEm = Em-Er
delEh = Eh-Er
tauHmax = 300

def zinf(U, Kz, Sz, Ez):
    return 1/(1+Kz*np.exp(Sz*(Ez-U)))

def tauz(U, tauzmax, Kz, Sz, Ez):
    return tauzmax*zinf(U, Kz, Sz, Ez)*np.sqrt(Kz*np.exp(Sz*(Ez-U)))

def minf(U):
    return zinf(U, Km, S, delEm)

def hinf(U):
    return zinf(U, Kh, -S, delEh)


Gna = Gm*R/(zinf(R, Km, S, delEm)*zinf(R, Kh, -S, delEh)*(delEna-R))

def cpg(numpy,backend):
    if numpy:
        g_ion = np.array([Gna])
        e_ion = np.array([delEna])

        pow_m = np.array([1])
        k_m = np.array([Km])
        slope_m = np.array([S])
        e_m = np.array([delEm])

        pow_h = np.array([1])
        k_h = np.array([Kh])
        slope_h = np.array([-S])
        e_h = np.array([delEh])
        tau_max_h = np.array([tauHmax])
    else:
        g_ion = torch.tensor([Gna])
        e_ion = torch.tensor([delEna])

        pow_m = torch.tensor([1])
        k_m = torch.tensor([Km])
        slope_m = torch.tensor([S])
        e_m = torch.tensor([delEm])

        pow_h = torch.tensor([1])
        k_h = torch.tensor([Kh])
        slope_h = torch.tensor([-S])
        e_h = torch.tensor([delEh])
        tau_max_h = torch.tensor([tauHmax])

    neuron_cpg = NonSpikingNeuronWithPersistentSodiumChannel(membrane_capacitance=Cm, membrane_conductance=Gm,
                                                             g_ion=g_ion,e_ion=e_ion,
                                                             k_m=k_m,slope_m=slope_m,e_m=e_m,
                                                             k_h=k_h,slope_h=slope_h,e_h=e_h,tau_max_h=tau_max_h,
                                                             name='HC',color='orange')

    neuron = NonSpikingNeuron()

    Ein = -100
    delEsyn = Ein - R

    gSyn = (-delta - delta * Gna * minf(delta) * hinf(delta) + Gna * minf(delta) * hinf(delta) * delEna) / (
                delta - delEsyn)

    synapse_cpg = NonSpikingSynapse(max_conductance=gSyn, relative_reversal_potential=delEsyn)

    net = Network()
    net.add_neuron(neuron_cpg, name='HC0', color='blue')
    net.add_input('HC0')
    net.add_output('HC0')
    net.add_neuron(neuron_cpg, name='HC1', color='orange')
    net.add_output('HC1')
    net.add_connection(synapse_cpg, 'HC0', 'HC1')
    net.add_connection(synapse_cpg, 'HC1', 'HC0')
    net.add_neuron(neuron)
    net.add_input(2)
    net.add_output(2)
    net.add_neuron(neuron)
    net.add_output(3)
    net.add_connection(NonSpikingSynapse(),2,3)

    I = 0
    tStart = 1000
    tEnd = 4000

    dt = 1
    tMax = 5000

    t = np.arange(0, tMax, dt)
    numSteps = np.size(t)

    if numpy:
        Iapp = np.zeros([numSteps,2])
        Ipert = np.zeros([numSteps,2])
    else:
        Iapp = torch.zeros([numSteps, 2])
        Ipert = torch.zeros([numSteps, 2])
    Iapp[tStart:tEnd, :] = I
    Ipert[1, 0] = 1
    Ipert[:,1] = 20
    if numpy:
        model = net.compile(backend=backend, dt=dt)
        data = np.zeros([len(t), net.get_num_outputs_actual()])
    else:
        model = net.compile(backend=backend, dt=dt,device='cpu')
        data = torch.zeros([len(t), net.get_num_outputs_actual()])
    inputs = Iapp + Ipert

    for i in range(len(t)):
        data[i] = model(inputs[i])
    if numpy:
        return data.transpose()
    else:
        return data.transpose(0,1)


print('Running Network 5:')
print('5: Numpy')
dataNumpy = cpg(True,'numpy')
print('5: Torch')
dataTorch = cpg(False,'torch')
print('5: Sparse')
dataSparse = cpg(False,'sparse')
print('5: Manual')
dataManual = cpg(True,'manual')

plt.figure()
plt.subplot(2,1,1)
plt.plot(t,dataNumpy[:][0],label='HC0',color='C0')
plt.plot(t,dataTorch[:][0],label='HC0',color='C1')
plt.plot(t,dataSparse[:][0],label='HC0',color='C2')
plt.plot(t,dataManual[:][0],label='HC0',color='C3')
plt.plot(t,dataNumpy[:][1],label='HC0',color='C0',linestyle='--')
plt.plot(t,dataTorch[:][1],label='HC0',color='C1',linestyle='--')
plt.plot(t,dataSparse[:][1],label='HC0',color='C2',linestyle='--')
plt.plot(t,dataManual[:][1],label='HC0',color='C3',linestyle='--')
plt.xlabel('t (ms)')
plt.ylabel('U (mV)')
title = 'CPG: Delta = ' + str(delta)

plt.subplot(2,1,2)
plt.plot(t,dataNumpy[:][2],label='HC0',color='C0')
plt.plot(t,dataTorch[:][2],label='HC0',color='C1')
plt.plot(t,dataSparse[:][2],label='HC0',color='C2')
plt.plot(t,dataManual[:][2],label='HC0',color='C3')
plt.plot(t,dataNumpy[:][3],label='HC0',color='C0',linestyle='--')
plt.plot(t,dataTorch[:][3],label='HC0',color='C1',linestyle='--')
plt.plot(t,dataSparse[:][3],label='HC0',color='C2',linestyle='--')
plt.plot(t,dataManual[:][3],label='HC0',color='C3',linestyle='--')

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Network 6: Resetting Networks
"""
def reset(numpy, backend, sparse=False):
    net = Network()
    net.add_neuron(NonSpikingNeuron())
    net.add_input(0)
    net.add_output(0)

    Cm = 5
    Gm = 1
    Ena = 50
    Er = -60
    R = 20

    S = 0.05
    delEna = Ena - Er
    Km = 1
    Kh = 0.5
    Em = -40
    Eh = -60
    delEm = Em - Er
    delEh = Eh - Er
    tauHmax = 300

    if numpy:
        g_ion = np.array([Gna])
        e_ion = np.array([delEna])

        pow_a = np.array([1])
        k_a = np.array([Km])
        slope_a = np.array([S])
        e_a = np.array([delEm])

        pow_b = np.array([1])
        k_b = np.array([Kh])
        slope_b = np.array([-S])
        e_b = np.array([delEh])
        tau_max_b = np.array([tauHmax])

        pow_c = np.array([0])
        k_c = np.array([1])
        slope_c = np.array([0])
        e_c = np.array([0])
        tau_max_c = np.array([1])
    else:
        g_ion = torch.tensor([Gna])
        e_ion = torch.tensor([delEna])

        pow_a = torch.tensor([1])
        k_a = torch.tensor([Km])
        slope_a = torch.tensor([S])
        e_a = torch.tensor([delEm])

        pow_b = torch.tensor([1])
        k_b = torch.tensor([Kh])
        slope_b = torch.tensor([-S])
        e_b = torch.tensor([delEh])
        tau_max_b = torch.tensor([tauHmax])

        pow_c = torch.tensor([0])
        k_c = torch.tensor([1])
        slope_c = torch.tensor([0])
        e_c = torch.tensor([0])
        tau_max_c = torch.tensor([1])

    neuron_gated = NonSpikingNeuronWithGatedChannels(membrane_capacitance=Cm, membrane_conductance=Gm, g_ion=g_ion,
                                                     e_ion=e_ion,
                                                     pow_a=pow_a, k_a=k_a, slope_a=slope_a, e_a=e_a,
                                                     pow_b=pow_b, k_b=k_b, slope_b=slope_b, e_b=e_b,
                                                     tau_max_b=tau_max_b,
                                                     pow_c=pow_c, k_c=k_c, slope_c=slope_c, e_c=e_c,
                                                     tau_max_c=tau_max_c,
                                                     name='HC', color='orange')

    net.add_neuron(neuron_gated)
    net.add_input(1)
    net.add_output(1)
    net.add_neuron(SpikingNeuron())
    net.add_input(2)
    net.add_output(2)

    I = 0


    dt = 1
    tMax = 5000

    t = np.arange(0, tMax, dt)
    numSteps = np.size(t)

    if numpy:
        Ipert = np.zeros([numSteps, 3])
    else:
        Ipert = torch.zeros([numSteps, 3])
    Ipert[1, 1] = 1
    Ipert[int(0.6*numSteps),1] = 1
    Ipert[int(0.3 * numSteps), 1] = 1
    Ipert[:, 0] = 20
    Ipert[:, 2] = 5
    if numpy:
        model = net.compile(backend=backend, dt=dt)
        data = np.zeros([len(t), net.get_num_outputs_actual()])
    else:
        model = net.compile(backend=backend, dt=dt, device='cpu')
        data = torch.zeros([len(t), net.get_num_outputs_actual()])

    if numpy:
        u = np.array([5.0,10.0,15.0])
        theta = np.array([sys.float_info.max,sys.float_info.max,10.0])
        b_gate = np.array([[0.0,0.0,5.0]])
        c_gate = np.array([[0.0, 0.0, 5.0]])
    else:
        if sparse:
            u = torch.tensor([5, 10, 15])
            theta = torch.tensor([np.inf, np.inf, 10])
            b_gate = torch.tensor([[0, 0, 5]]).to_sparse()
            c_gate = torch.tensor([[0, 0, 5]]).to_sparse()
        else:
            u = torch.tensor([5, 10, 15])
            theta = torch.tensor([np.inf, np.inf, 10])
            b_gate = torch.tensor([[0, 0, 5]])
            c_gate = torch.tensor([[0, 0, 5]])

    for i in range(numSteps):
        if i == (int(0.5 * numSteps)-1):
            model.reset()
        else:
            data[i] = model(Ipert[i,:])
    if numpy:
        return data.transpose(), t
    else:
        return data.transpose(0, 1), t

print('Running Network 6:')
print('6: Numpy')
dataNumpy, t = reset(True,'numpy')
print('5: Torch')
dataTorch, t = reset(False,'torch')
print('5: Sparse')
dataSparse, t = reset(False,'sparse',sparse=True)
print('5: Manual')
dataManual, t = reset(True,'manual')

plt.figure()
plt.subplot(3,1,1)
plt.plot(t,dataNumpy[:][0],label='Numpy',color='C0')
plt.plot(t,dataTorch[:][0],label='Torch',color='C1')
plt.plot(t,dataSparse[:][0],label='Sparse',color='C2')
plt.plot(t,dataManual[:][0],label='Manual',color='C3')
plt.legend()

plt.subplot(3,1,2)
plt.plot(t,dataNumpy[:][1],label='Numpy',color='C0')
plt.plot(t,dataTorch[:][1],label='Torch',color='C1')
plt.plot(t,dataSparse[:][1],label='Sparse',color='C2')
plt.plot(t,dataManual[:][1],label='Manual',color='C3')
plt.legend()

plt.subplot(3,1,3)
plt.plot(t,dataNumpy[:][2],label='Numpy',color='C0')
plt.plot(t,dataTorch[:][2],label='Torch',color='C1')
plt.plot(t,dataSparse[:][2],label='Sparse',color='C2')
plt.plot(t,dataManual[:][2],label='Manual',color='C3')
plt.legend()

plt.figure()
plt.subplot(3,1,1)
plt.title('Error between other backends and manual')
plt.plot(t,dataNumpy[:][0]-dataManual[:][0],label='Numpy',color='C0')
plt.plot(t,dataTorch[:][0]-dataManual[:][0],label='Torch',color='C1')
plt.plot(t,dataSparse[:][0]-dataManual[:][0],label='Sparse',color='C2')
plt.legend()

plt.subplot(3,1,2)
plt.plot(t,dataNumpy[:][1]-dataManual[:][1],label='Numpy',color='C0')
plt.plot(t,dataTorch[:][1]-dataManual[:][1],label='Torch',color='C1')
plt.plot(t,dataSparse[:][1]-dataManual[:][1],label='Sparse',color='C2')
plt.legend()

plt.subplot(3,1,3)
plt.plot(t,dataNumpy[:][2]-dataManual[:][2],label='Numpy',color='C0')
plt.plot(t,dataTorch[:][2]-dataManual[:][2],label='Torch',color='C1')
plt.plot(t,dataSparse[:][2]-dataManual[:][2],label='Sparse',color='C2')
plt.legend()

# """
# Network 7: Loading network
# """
#
# neuron_non_spike = NonSpikingNeuron()
# neuron_spike = SpikingNeuron()
# neuron_gated = NonSpikingNeuronWithPersistentSodiumChannel()
# neuron_gated_torch = NonSpikingNeuronWithPersistentSodiumChannel(g_ion=torch.tensor([1.0485070729908987],device='cpu'),
#                                                                  e_ion=torch.tensor([110],device='cpu'),
#                                                                  k_m=torch.tensor([1],device='cpu'),
#                                                                  slope_m=torch.tensor([0.05],device='cpu'),
#                                                                  e_m=torch.tensor([20],device='cpu'),
#                                                                  k_h=torch.tensor([0.5],device='cpu'),
#                                                                  slope_h=torch.tensor([-0.05],device='cpu'),
#                                                                  e_h=torch.tensor([0],device='cpu'),
#                                                                  tau_max_h=torch.tensor([300],device='cpu')
#                                                                  )
#
# synapse_non_spike = NonSpikingSynapse()
# synapse_spike = SpikingSynapse()
# synapse_spike_delay = SpikingSynapse(transmission_delay=5)
# synapse_elect = ElectricalSynapse(conductance=1)
# synapse_elect_rect = ElectricalSynapse(conductance=1,rect=True)
#
# net = Network('Tutorial 9 Network')
# net.add_neuron(neuron_non_spike,name='0',color='blue')
# net.add_neuron(neuron_non_spike,name='1',color='orange')
# net.add_neuron(neuron_spike,name='2',color='green')
# net.add_neuron(neuron_spike,name='3',color='red')
# net.add_neuron(neuron_gated,name='4',color='purple')
# net.add_neuron(neuron_non_spike,name='5',color='brown')
# net.add_neuron(neuron_non_spike,name='6',color='pink',initial_value=5)
# net.add_neuron(neuron_spike, name='7',color='gray')
#
# net.add_input('0')
# net.add_input('2')
# net.add_input('4')
#
# net.add_output('0')
# net.add_output('1')
# net.add_output('2')
# net.add_output('3')
# net.add_output('4')
# net.add_output('5')
# net.add_output('6')
# net.add_output('7')
#
# net.add_connection(synapse_non_spike,'0','1')
# net.add_connection(synapse_spike,'2','3')
# net.add_connection(synapse_elect,'0','5')
# net.add_connection(synapse_elect_rect,'6','5')
# net.add_connection(synapse_spike_delay,'2','3')
#
# netTorch = Network('Tutorial 9 Network')
# netTorch.add_neuron(neuron_non_spike,name='0',color='blue')
# netTorch.add_neuron(neuron_non_spike,name='1',color='orange')
# netTorch.add_neuron(neuron_spike,name='2',color='green')
# netTorch.add_neuron(neuron_spike,name='3',color='red')
# netTorch.add_neuron(neuron_gated_torch,name='4',color='purple')
# netTorch.add_neuron(neuron_non_spike,name='5',color='brown')
# netTorch.add_neuron(neuron_non_spike,name='6',color='pink',initial_value=5)
# netTorch.add_neuron(neuron_spike, name='7',color='gray')
# netTorch.add_input('0')
# netTorch.add_input('2')
# netTorch.add_input('4')
# netTorch.add_output('0')
# netTorch.add_output('1')
# netTorch.add_output('2')
# netTorch.add_output('3')
# netTorch.add_output('4')
# netTorch.add_output('5')
# netTorch.add_output('6')
# netTorch.add_output('7')
# netTorch.add_connection(synapse_non_spike,'0','1')
# netTorch.add_connection(synapse_spike,'2','3')
# netTorch.add_connection(synapse_elect,'0','5')
# netTorch.add_connection(synapse_elect_rect,'6','5')
# netTorch.add_connection(synapse_spike_delay,'2','3')
#
# plt.figure()
# # net.render_graph(view=True)
# dt = 0.01
# t_max = 50
#
# # Initialize a vector of timesteps
# t = np.arange(0, t_max, dt)
#
# print('Running Network 7:')
# print('7: Numpy')
# # Initialize vectors which store the input to our network, and for data to be written to during simulation from outputs
# inputs = np.zeros([len(t),net.get_num_inputs()])+5.0 # Input vector must be 2d, even if second dimension is 1
# data_original = np.zeros([len(t),net.get_num_outputs_actual()])
# data_load = np.zeros([len(t),net.get_num_outputs_actual()])
#
# model_original = net.compile(backend='numpy',dt=dt)
# model_original.save(filename='saveNumpy.p')
#
# model_load = sns_toolbox.simulate.backends.SNS_Numpy('saveNumpy.p',dt=dt)
#
# """Simulate the network"""
# # At every step, apply the current input to a forward pass of the network and store the results in 'data'
# for i in range(len(t)):
#     data_original[i,:] = model_original(inputs[i,:])
#     data_load[i, :] = model_load(inputs[i, :])
# data_original = data_original.transpose()
# data_load = data_load.transpose()
#
# plt.subplot(2,2,1)
# plt.plot(t,data_original[:][0] - data_load[:][0],label='0',color='C0')
# plt.plot(t,data_original[:][1] - data_load[:][1],label='1',color='C1')
# plt.plot(t,data_original[:][2] - data_load[:][2],label='2',color='C2')
# plt.plot(t,data_original[:][3] - data_load[:][3],label='3',color='C3')
# plt.plot(t,data_original[:][4] - data_load[:][4],label='4',color='C4')
# plt.plot(t,data_original[:][5] - data_load[:][5],label='5',color='C5')
# plt.plot(t,data_original[:][6] - data_load[:][6],label='6',color='C6')
# plt.plot(t,data_original[:][7] - data_load[:][7],label='7',color='C7')
# plt.legend()
#
# print('7: Torch')
# # Initialize vectors which store the input to our network, and for data to be written to during simulation from outputs
# inputs = torch.zeros([len(t),net.get_num_inputs()],device='cpu')+5.0 # Input vector must be 2d, even if second dimension is 1
# data_original = torch.zeros([len(t),net.get_num_outputs_actual()],device='cpu')
# data_load = torch.zeros([len(t),net.get_num_outputs_actual()],device='cpu')
#
# model_original = net.compile(backend='torch'Torch,dt=dt,device='cpu')
# model_original.save(filename='saveTorch.p')
#
# model_load = sns_toolbox.simulate.backends.SNS_Torch('saveTorch.p',dt=dt,device='cpu')
#
# """Simulate the network"""
# # At every step, apply the current input to a forward pass of the network and store the results in 'data'
# for i in range(len(t)):
#     data_original[i,:] = model_original(inputs[i,:])
#     data_load[i, :] = model_load(inputs[i, :])
# data_original = data_original.transpose(0,1)
# data_load = data_load.transpose(0,1)
#
# plt.subplot(2,2,2)
# plt.plot(t,data_original[:][0] - data_load[:][0],label='0',color='C0')
# plt.plot(t,data_original[:][1] - data_load[:][1],label='1',color='C1')
# plt.plot(t,data_original[:][2] - data_load[:][2],label='2',color='C2')
# plt.plot(t,data_original[:][3] - data_load[:][3],label='3',color='C3')
# plt.plot(t,data_original[:][4] - data_load[:][4],label='4',color='C4')
# plt.plot(t,data_original[:][5] - data_load[:][5],label='5',color='C5')
# plt.plot(t,data_original[:][6] - data_load[:][6],label='6',color='C6')
# plt.plot(t,data_original[:][7] - data_load[:][7],label='7',color='C7')
# plt.legend()
#
# print('7: Sparse')
# # Initialize vectors which store the input to our network, and for data to be written to during simulation from outputs
# inputs = torch.zeros([len(t),net.get_num_inputs()],device='cpu')+5.0 # Input vector must be 2d, even if second dimension is 1
# data_original = torch.zeros([len(t),net.get_num_outputs_actual()],device='cpu')
# data_load = torch.zeros([len(t),net.get_num_outputs_actual()],device='cpu')
#
# model_original = net.compile(backend='sparse'Torch,dt=dt,device='cpu')
# model_original.save(filename='saveSparse.p')
#
# model_load = sns_toolbox.simulate.backends.SNS_Sparse('saveSparse.p',dt=dt,device='cpu')
#
# """Simulate the network"""
# # At every step, apply the current input to a forward pass of the network and store the results in 'data'
# for i in range(len(t)):
#     data_original[i,:] = model_original(inputs[i,:])
#     data_load[i, :] = model_load(inputs[i, :])
# data_original = data_original.transpose(0,1)
# data_load = data_load.transpose(0,1)
#
# plt.subplot(2,2,3)
# plt.plot(t,data_original[:][0] - data_load[:][0],label='0',color='C0')
# plt.plot(t,data_original[:][1] - data_load[:][1],label='1',color='C1')
# plt.plot(t,data_original[:][2] - data_load[:][2],label='2',color='C2')
# plt.plot(t,data_original[:][3] - data_load[:][3],label='3',color='C3')
# plt.plot(t,data_original[:][4] - data_load[:][4],label='4',color='C4')
# plt.plot(t,data_original[:][5] - data_load[:][5],label='5',color='C5')
# plt.plot(t,data_original[:][6] - data_load[:][6],label='6',color='C6')
# plt.plot(t,data_original[:][7] - data_load[:][7],label='7',color='C7')
# plt.legend()
#
# print('7: Manual')
# # Initialize vectors which store the input to our network, and for data to be written to during simulation from outputs
# inputs = np.zeros([len(t),net.get_num_inputs()])+5.0 # Input vector must be 2d, even if second dimension is 1
# data_original = np.zeros([len(t),net.get_num_outputs_actual()])
# data_load = np.zeros([len(t),net.get_num_outputs_actual()])
#
# model_original = net.compile(backend='manual',dt=dt)
# model_original.save(filename='saveManual.p')
#
# model_load = sns_toolbox.simulate.backends.SNS_Manual('saveManual.p',dt=dt)
#
# """Simulate the network"""
# # At every step, apply the current input to a forward pass of the network and store the results in 'data'
# for i in range(len(t)):
#     data_original[i,:] = model_original(inputs[i,:])
#     data_load[i, :] = model_load(inputs[i, :])
# data_original = data_original.transpose()
# data_load = data_load.transpose()
#
# plt.subplot(2,2,4)
# plt.plot(t,data_original[:][0] - data_load[:][0],label='0',color='C0')
# plt.plot(t,data_original[:][1] - data_load[:][1],label='1',color='C1')
# plt.plot(t,data_original[:][2] - data_load[:][2],label='2',color='C2')
# plt.plot(t,data_original[:][3] - data_load[:][3],label='3',color='C3')
# plt.plot(t,data_original[:][4] - data_load[:][4],label='4',color='C4')
# plt.plot(t,data_original[:][5] - data_load[:][5],label='5',color='C5')
# plt.plot(t,data_original[:][6] - data_load[:][6],label='6',color='C6')
# plt.plot(t,data_original[:][7] - data_load[:][7],label='7',color='C7')
# plt.legend()

plt.show()