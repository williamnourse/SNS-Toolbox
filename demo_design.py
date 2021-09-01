"""
Example where a network is designed, compiled, and run
William Nourse
August 31, 2021
Everything is priced in
"""
import matplotlib.pyplot as plt
import numpy as np

from sns_toolbox.design.neurons import NonSpikingNeuron, SpikingNeuron
from sns_toolbox.design.connections import NonSpikingSynapse
from sns_toolbox.design.networks import Network
from sns_toolbox.simulate.backends import SNS_Numpy

"""
########################################################################################################################
DESIGN
"""
# Neuron and Synapse Types
nonSpike = NonSpikingNeuron(name='NonSpiking', color='antiquewhite')
spike0 = SpikingNeuron(name='m=0', color='aqua')
spikeL0 = SpikingNeuron(name='m<0',thresholdProportionalityConstant=-1,color='aquamarine')
spikeG0 = SpikingNeuron(name='m>0',thresholdProportionalityConstant=1,color='azure')
nonExcite = NonSpikingSynapse(name='Excitatory NonSpiking')
nonInhibit = NonSpikingSynapse(name='Inhibitory NonSpiking',relativeReversalPotential=-40.0)
nonMod = NonSpikingSynapse(name='Modulatory NonSpiking',relativeReversalPotential=0)

# Network with different m values
netVaryM = Network(name='Different Ms')
netVaryM.addNeuron(spike0)
netVaryM.addNeuron(spikeL0)
netVaryM.addNeuron(spikeG0)
netVaryM.addInput(name='I0', color='brown')
netVaryM.addInputConnection(1, 0, 0)
netVaryM.addInputConnection(1, 0, 1)
netVaryM.addInputConnection(1, 0, 2)
netVaryM.addOutput(name='O0V', color='cadetblue')
netVaryM.addOutput(name='O1S', color='chartreuse', spiking=True)
netVaryM.addOutput(name='O2V', color='cadetblue')
netVaryM.addOutput(name='O3S', color='chartreuse', spiking=True)
netVaryM.addOutput(name='O4V', color='cadetblue')
netVaryM.addOutput(name='O5S', color='chartreuse', spiking=True)
netVaryM.addOutputConnection(1, 0, 0)
netVaryM.addOutputConnection(1, 0, 1)
netVaryM.addOutputConnection(1, 1, 2)
netVaryM.addOutputConnection(1, 1, 3)
netVaryM.addOutputConnection(1, 2, 4)
netVaryM.addOutputConnection(1, 2, 5)
netVaryM.renderGraph(view=False)

# Basic NonSpiking Neurons
netNonSpike = Network(name='Simple NonSpiking')
netNonSpike.addInput(name='I1',color='chocolate')
netNonSpike.addNeuron(nonSpike,name='NS0')
netNonSpike.addNeuron(nonSpike,name='NS1',color='coral')
netNonSpike.addNeuron(nonSpike,name='NS2',color='cornflowerblue')
netNonSpike.addNeuron(nonSpike,name='NS3',color='cornsilk')
netNonSpike.addOutput(name='O6V',color='crimson')
netNonSpike.addOutput(name='O7V',color='cyan')
netNonSpike.addOutput(name='O8V',color='darkblue')
netNonSpike.addOutput(name='O9V',color='darkcyan')
netNonSpike.addInputConnection(2,0,0)
netNonSpike.addInputConnection(2,0,1)
netNonSpike.addInputConnection(2,0,2)
netNonSpike.addSynapse(nonMod,1,0)
netNonSpike.addSynapse(nonInhibit,1,2)
netNonSpike.addSynapse(nonExcite,1,3)
netNonSpike.addOutputConnection(1,0,0)
netNonSpike.addOutputConnection(1,1,1)
netNonSpike.addOutputConnection(1,2,2)
netNonSpike.addOutputConnection(1,3,3)
netNonSpike.renderGraph(view=False)

# Network which will be simulated, containing all other networks
totalNet = Network(name='Total Network')
totalNet.addNetwork(netVaryM, color='blueviolet')
totalNet.addNetwork(netNonSpike,color='darkgoldenrod')
totalNet.renderGraph(view=True)

"""
########################################################################################################################
SIMULATION
"""

dt = 0.01
model = SNS_Numpy(totalNet,dt=dt)
tMax = 100
t = np.arange(0,tMax,dt)
inputs = np.zeros([len(t),totalNet.getNumInputs()])+10
data = np.zeros([len(t),totalNet.getNumOutputs()])
for i in range(len(t)):
    data[i][:] = model.forward(inputs[i][:])

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
plt.plot(t,data.transpose()[:][6],label='Mod')
plt.plot(t,data.transpose()[:][7],label='Source')
plt.plot(t,data.transpose()[:][8],label='Inhibit')
plt.plot(t,data.transpose()[:][9],label='Excite')
plt.legend()

plt.show()
