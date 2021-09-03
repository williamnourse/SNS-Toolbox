"""
Example where a network is designed, compiled, and run
William Nourse
August 31, 2021
Everything is priced in
"""
import matplotlib.pyplot as plt
import numpy as np

from sns_toolbox.design.neurons import NonSpikingNeuron, SpikingNeuron
from sns_toolbox.design.connections import NonSpikingSynapse, SpikingSynapse
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
spikeExcite = SpikingSynapse(name='Excitatory Spiking')

# Network with different m values
netVaryM = Network(name='Different Ms')
netVaryM.addNeuron(spike0,name='m=0')
netVaryM.addNeuron(spikeL0,name='m<0')
netVaryM.addNeuron(spikeG0,name='m>0')
netVaryM.addInput(name='I0', color='brown')
netVaryM.addInputConnection(1, 'I0', 'm=0')
netVaryM.addInputConnection(1, 'I0', 'm<0')
netVaryM.addInputConnection(1, 'I0', 'm>0')
netVaryM.addOutput('m=0',name='O0V', color='cadetblue')
netVaryM.addOutput('m=0',name='O1S', color='chartreuse', spiking=True)
netVaryM.addOutput('m<0',name='O2V', color='cadetblue')
netVaryM.addOutput('m<0',name='O3S', color='chartreuse', spiking=True)
netVaryM.addOutput('m>0',name='O4V', color='cadetblue')
netVaryM.addOutput('m>0',name='O5S', color='chartreuse', spiking=True)
netVaryM.renderGraph(view=False)

# Basic NonSpiking Neurons
netNonSpike = Network(name='Simple NonSpiking')
netNonSpike.addInput(name='I1',color='chocolate')
netNonSpike.addNeuron(nonSpike,name='NS0')
netNonSpike.addNeuron(nonSpike,name='NS1',color='coral')
netNonSpike.addNeuron(nonSpike,name='NS2',color='cornflowerblue')
netNonSpike.addNeuron(nonSpike,name='NS3',color='cornsilk')
netNonSpike.addOutput(0,name='O6V',color='crimson')
netNonSpike.addOutput(1,name='O7V',color='cyan')
netNonSpike.addOutput(2,name='O8V',color='darkblue')
netNonSpike.addOutput(3,name='O9V',color='darkcyan')
netNonSpike.addInputConnection(2,0,0)
netNonSpike.addInputConnection(2,0,1)
netNonSpike.addInputConnection(2,0,2)
netNonSpike.addSynapse(nonMod,1,0)
netNonSpike.addSynapse(nonInhibit,1,2)
netNonSpike.addSynapse(nonExcite,1,3)
netNonSpike.renderGraph(view=False)

# Mixing Populations
netPop = Network(name='Mixed Populations')
netPop.addInput(name='I2',color='darkgoldenrod')
netPop.addPopulation(spike0,numNeurons=3,name='S3',color='darkmagenta')
netPop.addPopulation(nonSpike,numNeurons=4,name='NS4',color='darkolivegreen')
netPop.addPopulation(spike0,numNeurons=2,name='S2',color='darkorange')
netPop.addOutput(0,name='O10V',color='darkorchid')
netPop.addOutput(0,name='O11S',spiking=True,color='darkred')
netPop.addOutput(2,name='O12S',color='darksalmon')
netPop.addOutput(2,name='O13S',spiking=True,color='darkseagreen')
netPop.addOutput(1,name='O14V',color='darkslateblue')
netPop.addInputConnection(1,0,0)
netPop.addInputConnection(1,0,1)
netPop.addSynapse(spikeExcite,0,2)
netPop.addSynapse(nonMod,1,2)
netPop.renderGraph(view=False)

# Network which will be simulated, containing all other networks
totalNet = Network(name='Total Network')
totalNet.addNetwork(netVaryM, color='blueviolet')
totalNet.addNetwork(netNonSpike,color='darkgoldenrod')
totalNet.addNetwork(netPop,color='darkslategrey')
totalNet.renderGraph(view=True)

"""
########################################################################################################################
SIMULATION
"""

dt = 0.01
model = SNS_Numpy(totalNet,dt=dt,debug=False)
tMax = 100
t = np.arange(0,tMax,dt)
inputs = np.zeros([len(t),totalNet.getNumInputs()])+10
data = np.zeros([len(t),totalNet.getNumOutputsActual()])
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
