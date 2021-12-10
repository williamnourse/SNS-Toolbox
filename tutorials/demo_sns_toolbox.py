"""
Example where a network is designed, compiled, and run
William Nourse
August 31, 2021
Everything is priced in
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import time

from sns_toolbox.design.neurons import NonSpikingNeuron, SpikingNeuron
from sns_toolbox.design.connections import NonSpikingSynapse, SpikingSynapse
from sns_toolbox.design.networks import Network
from sns_toolbox.simulate.backends import SNS_Numpy, SNS_Torch, SNS_Torch_Large

"""
########################################################################################################################
DESIGN
"""
render = False

# Neuron and Synapse Types
nonSpike = NonSpikingNeuron(name='NonSpiking', color='antiquewhite')
spike0 = SpikingNeuron(name='m=0', color='aqua')
spikeL0 = SpikingNeuron(name='m<0', threshold_proportionality_constant=-1, color='aquamarine')
spikeG0 = SpikingNeuron(name='m>0', threshold_proportionality_constant=1, color='azure')
nonExcite = NonSpikingSynapse(name='Excitatory NonSpiking')
nonInhibit = NonSpikingSynapse(name='Inhibitory NonSpiking', relative_reversal_potential=-40.0)
nonMod = NonSpikingSynapse(name='Modulatory NonSpiking', relative_reversal_potential=0)
spikeExcite = SpikingSynapse(name='Excitatory Spiking')

# Network with different m values
netVaryM = Network(name='Different Ms')
netVaryM.add_neuron(spike0, name='m=0')
netVaryM.add_neuron(spikeL0, name='m<0')
netVaryM.add_neuron(spikeG0, name='m>0')
netVaryM.add_input(name='I0', color='brown')
netVaryM.addInputConnection(1, 'I0', 'm=0')
netVaryM.addInputConnection(1, 'I0', 'm<0')
netVaryM.addInputConnection(1, 'I0', 'm>0')
netVaryM.add_output('m=0', name='O0V', color='cadetblue')
netVaryM.add_output('m=0', name='O1S', color='chartreuse', spiking=True)
netVaryM.add_output('m<0', name='O2V', color='cadetblue')
netVaryM.add_output('m<0', name='O3S', color='chartreuse', spiking=True)
netVaryM.add_output('m>0', name='O4V', color='cadetblue')
netVaryM.add_output('m>0', name='O5S', color='chartreuse', spiking=True)
netVaryM.render_graph(view=render)

# Basic NonSpiking Neurons
netNonSpike = Network(name='Simple NonSpiking')
netNonSpike.add_input(name='I1', color='chocolate')
netNonSpike.add_neuron(nonSpike, name='NS0')
netNonSpike.add_neuron(nonSpike, name='NS1', color='coral')
netNonSpike.add_neuron(nonSpike, name='NS2', color='cornflowerblue')
netNonSpike.add_neuron(nonSpike, name='NS3', color='cornsilk')
netNonSpike.add_output(0, name='O6V', color='crimson')
netNonSpike.add_output(1, name='O7V', color='cyan')
netNonSpike.add_output(2, name='O8V', color='darkblue')
netNonSpike.add_output(3, name='O9V', color='darkcyan')
netNonSpike.addInputConnection(2,0,0)
netNonSpike.addInputConnection(2,0,1)
netNonSpike.addInputConnection(2,0,2)
netNonSpike.add_synapse(nonMod, 1, 0)
netNonSpike.add_synapse(nonInhibit, 1, 2)
netNonSpike.add_synapse(nonExcite, 1, 3)
netNonSpike.render_graph(view=render)

# Mixing Populations
netPop = Network(name='Mixed Populations')
netPop.add_input(name='I2', color='darkgoldenrod')
netPop.add_population(spike0, num_neurons=3, name='S3', color='darkmagenta')
netPop.add_population(nonSpike, num_neurons=4, name='NS4', color='darkolivegreen')
netPop.add_population(spike0, num_neurons=2, name='S2', color='darkorange')
netPop.add_output(0, name='O10V', color='darkorchid')
netPop.add_output(0, name='O11S', spiking=True, color='darkred')
netPop.add_output(2, name='O12S', color='darksalmon')
netPop.add_output(2, name='O13S', spiking=True, color='darkseagreen')
netPop.add_output(1, name='O14V', color='darkslateblue')
netPop.addInputConnection(1,0,0)
netPop.addInputConnection(1,0,1)
netPop.add_synapse(spikeExcite, 0, 2)
netPop.add_synapse(nonMod, 1, 2)
netPop.render_graph(view=render)

# Network which will be simulated, containing all other networks
totalNet = Network(name='Total Network')
totalNet.add_network(netVaryM, color='blueviolet')
totalNet.add_network(netNonSpike, color='darkgoldenrod')
totalNet.add_network(netPop, color='darkslategrey')
totalNet.render_graph(view=render)

"""
########################################################################################################################
SIMULATION
"""

dt = 0.01
tMax = 100
t = np.arange(0,tMax,dt)
inputs = np.zeros([len(t), totalNet.get_num_inputs()]) + 10
data = np.zeros([len(t), totalNet.get_num_outputs_actual()])
numpy = True
if numpy:
    model = SNS_Numpy(totalNet, dt=dt, debug=False)
else:
    device = 'cuda'
    print(time.time())
    model = SNS_Torch(totalNet,dt=dt,debug=False)
    print(time.time())
    inputs = torch.from_numpy(inputs).to(device)
    data = torch.from_numpy(data).to(device)
start = time.time()
for i in range(len(t)):
    data[i,:] = model.forward(inputs[i,:])
end = time.time()
print('%f sec'%(end-start))
if not numpy:
    data = data.cpu().numpy()
"""
########################################################################################################################
PLOTTING
"""
# First Net
plt.figure()
for i in range(3):
    plt.subplot(3,1,i+1)
    plt.plot(t,data.transpose()[:][2*i])
    plt.plot(t, data.transpose()[:][2*i+1])

# Second Net
plt.figure()
plt.title('NonSpiking Network')
plt.plot(t,data.transpose()[:][6],label='Mod')
plt.plot(t,data.transpose()[:][7],label='Source')
plt.plot(t,data.transpose()[:][8],label='Inhibit')
plt.plot(t,data.transpose()[:][9],label='Excite')
plt.legend()

# Third Net
plt.figure()
plt.subplot(5,1,1)
plt.plot(t,data.transpose()[:][10])
plt.plot(t,data.transpose()[:][11])
plt.plot(t,data.transpose()[:][12])
plt.title('Spiking Source Population Voltage')
plt.subplot(5,1,2)
plt.plot(t,data.transpose()[:][13])
plt.plot(t,data.transpose()[:][14])
plt.plot(t,data.transpose()[:][15])
plt.title('Spiking Source Population Spikes')
plt.subplot(5,1,3)
plt.plot(t,data.transpose()[:][16])
plt.plot(t,data.transpose()[:][17])
plt.title('Spiking Destination Population Voltage')
plt.subplot(5,1,4)
plt.plot(t,data.transpose()[:][18])
plt.plot(t,data.transpose()[:][19])
plt.title('Spiking Source Population Spikes')
plt.subplot(5,1,5)
plt.plot(t,data.transpose()[:][20])
plt.plot(t,data.transpose()[:][21])
plt.plot(t,data.transpose()[:][22])
plt.plot(t,data.transpose()[:][23])
plt.title('Spiking Source Population Voltage')

plt.show()
