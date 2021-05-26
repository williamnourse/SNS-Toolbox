"""
Simple demo file to make sure the backends work in general
William Nourse
May 26, 2021
$GME go BRRRRRRR
"""
import numpy as np
import matplotlib.pyplot as plt

from sns_toolbox.design.neurons import NonSpikingNeuron
from sns_toolbox.design.synapses import NonSpikingSynapse
from sns_toolbox.design.networks import NonSpikingNetwork
from sns_toolbox.simulate.backends import Manual

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
DESIGN
"""
# Simple network: 2 neurons with a transmission synapse between them
simple = NonSpikingNeuron()
transmit = NonSpikingSynapse(name='Transmit')
net = NonSpikingNetwork(name='Network')
net.addNeuron(simple,suffix='A')
net.addNeuron(simple,suffix='B')
net.addSynapse(transmit,0,1)
net.renderGraph(view=True)

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
BUILD
"""
dt = 1    # ms
appliedCurrents = np.array([20.0,0])
netManual = Manual(net,dt=dt)

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SIMULATION
"""

# stateLast = np.zeros(netManual.numNeurons)
tMax = 100  # ms
t = np.arange(0,tMax,dt)
numSteps = len(t)
Umanual = np.zeros([netManual.numNeurons,numSteps])

for i in range(1,numSteps):
    Umanual[:,i] = netManual.forward(Umanual[:,i-1],appliedCurrents)

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
PLOTTING
"""

plt.figure()
for i in range(netManual.numNeurons):
    plt.plot(t,Umanual[i,:],label=str(i))
plt.xlabel('t (ms)')
plt.ylabel('U (mV)')
plt.title('Manual Computation')
plt.legend()
plt.show()
