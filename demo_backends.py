"""
Simple demo file to make sure the backends work in general
William Nourse
May 26, 2021
$GME go BRRRRRRR
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, lil_matrix

from sns_toolbox.design.neurons import NonSpikingNeuron
from sns_toolbox.design.connections import NonSpikingSynapse
from sns_toolbox.design.networks import Network
from sns_toolbox.simulate.backends import SNS_Manual, SNS_SciPy

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
DESIGN
"""
# Simple network: 2 neurons with a transmission synapse between them
numNeurons = 2
simple = NonSpikingNeuron()
transmit = NonSpikingSynapse(name='Transmit')
net = Network(name='Network')
net.addNeuron(simple,suffix='A')
net.addNeuron(simple,suffix='B')
net.addSynapse(transmit,0,1)
net.renderGraph(view=True)

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
BUILD
"""
dt = 0.1    # ms
appliedCurrents = np.array([20.0,0])
appliedSparse = csr_matrix(appliedCurrents)
netManual = SNS_Manual(net, dt=dt)
netScipy = SNS_SciPy(net, dt=dt)
"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SIMULATION
"""

# stateLast = np.zeros(netManual.numNeurons)
tMax = 100  # ms
t = np.arange(0,tMax,dt)
numSteps = len(t)
Umanual = np.zeros([netManual.numNeurons,numSteps])
Uscipy = lil_matrix(np.zeros([numSteps,netScipy.numNeurons]))

for i in range(1,numSteps):
    Umanual[:,i] = netManual.forward(Umanual[:,i-1],appliedCurrents)
    Uscipy[i, :] = netScipy.forward(Uscipy[i-1, :],appliedSparse)
Uscipy = Uscipy.toarray()
"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
PLOTTING
"""

plt.figure()
for i in range(netManual.numNeurons):
    plt.plot(t,Umanual[i,:],label=str(i))
plt.xlabel('t (ms)')
plt.ylabel('U (mV)')
plt.title('SNS_Manual Computation')
plt.legend()

plt.figure()
for i in range(numNeurons):
    plt.plot(t,Uscipy[:,i].transpose(),label=str(i))
plt.xlabel('t (ms)')
plt.ylabel('U (mV)')
plt.title('Matrix Computation')
plt.legend()

# Difference
plt.figure()
for i in range(numNeurons):
    plt.plot(t,Umanual[i,:]-Uscipy[:,i].transpose(),label=str(i))
plt.xlabel('t (ms)')
plt.ylabel('U (mV)')
plt.title('SNS_Manual - Matrix')
plt.legend()

plt.show()
