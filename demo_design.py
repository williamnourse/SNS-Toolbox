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

nonSpike = NonSpikingNeuron(name='NonSpiking', color='antiquewhite')
spike0 = SpikingNeuron(name='m=0', color='aqua')
spikeL0 = SpikingNeuron(name='m<0',thresholdProportionalityConstant=-1,color='aquamarine')
spikeG0 = SpikingNeuron(name='m>0',thresholdProportionalityConstant=1,color='azure')

net0 = Network(name='Network0')

net0.addNeuron(spike0)
net0.addNeuron(spikeL0)
net0.addNeuron(spikeG0)

net0.addInput(name='I0', color='brown')
net0.addInputConnection(1, 0, 0)
net0.addInputConnection(1, 0, 1)
net0.addInputConnection(1, 0, 2)

net0.addOutput(name='O0V',color='cadetblue')

net0.addOutput(name='O1S',color='chartreuse',spiking=True)
net0.addOutput(name='O2V',color='cadetblue')
net0.addOutput(name='O3S',color='chartreuse',spiking=True)
net0.addOutput(name='O4V',color='cadetblue')
net0.addOutput(name='O5S',color='chartreuse',spiking=True)
net0.addOutputConnection(1,0,0)
net0.addOutputConnection(1,0,1)
net0.addOutputConnection(1,1,2)
net0.addOutputConnection(1,1,3)
net0.addOutputConnection(1,2,4)
net0.addOutputConnection(1,2,5)

net0.renderGraph(view=True)

totalNet = Network(name='Total Network')
totalNet.addNetwork(net0, color='blueviolet')
totalNet.renderGraph(view=True)

"""
########################################################################################################################
SIMULATION
"""

dt = 0.01
model = SNS_Numpy(totalNet,dt=dt)
tMax = 10
t = np.arange(0,tMax,dt)
inputs = np.zeros([len(t),totalNet.getNumInputs()])+10
data = np.zeros([len(t),totalNet.getNumOutputs()])
for i in range(len(t)):
    data[i][:] = model.forward(inputs[i][:])

"""
########################################################################################################################
PLOTTING
"""
plt.figure()
for i in range(3):
    plt.subplot(3,1,i+1)
    plt.plot(t,data.transpose()[:][2*i])
    plt.plot(t, data.transpose()[:][2*i+1])

plt.show()
