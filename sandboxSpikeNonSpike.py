"""
Let's prototype a simulator for spiking and non-spiking together using numpy
William Nourse
August 20th, 2021
I am... inevitable
"""

"""
########################################################################################################################
IMPORTS
"""

import numpy as np
import matplotlib.pyplot as plt

"""
########################################################################################################################
NETWORK CONSTRUCTION
"""

# Network Parameters
R = 20.0    # neural activity range (mV)
numNeurons = 11 # number of neurons in the demo network
Uini = np.zeros(numNeurons)     # all neurons should start with zero voltage
dt = 0.001  # timestep (ms)

# External Input Handling
externalCurrent = 20.0         # magnitude of applied current (nA)
inputNodes = [externalCurrent] # vector of input currents
inputConnectivityVector = np.array([[1,1,1,1,1,1,0,0,0,0,0]])   # this is a vector because input is 1d

# Initialize Neurons
Ulast = np.zeros(numNeurons) + Uini   # previous timestep
U = np.zeros(numNeurons)        # current timestep

# Neural Parameters
Cm = np.zeros(numNeurons) + 5.0 # membrane capacitance (nF)
Gm = np.zeros(numNeurons) + 1.0 # membrane conductance (uS)
Ib = np.zeros(numNeurons) + 0.0 # bias current (nA)

# Output Nodes
outputNodes = np.zeros(numNeurons)  # we will read the voltage from every neuron in the network
outputConnectivityMatrix = np.identity(numNeurons)  # matrix because we're reading every neural voltage

"""
########################################################################################################################
SIMULATION
"""
tmax = 50.0                 # max time (ms)
t = np.arange(0,tmax,dt)    # simulation time vector
numSteps = len(t)           # number of simulation steps
timeFactor = dt/Cm          # multiplicative time factor (save on operations)

outVals = np.zeros([numNeurons,numSteps])  # vector for storing output values

for i in range(numSteps):
    Ulast = np.copy(U)
    Iapp = inputConnectivityVector*inputNodes
    U = Ulast + timeFactor*(-Gm*Ulast+Ib+Iapp)
    outputNodes = np.matmul(U,outputConnectivityMatrix)
    outVals[:,i] = outputNodes

"""
########################################################################################################################
PLOTTING
"""

plt.figure()
for i in range(numNeurons):
    plt.subplot(numNeurons,1,i+1)
    plt.plot(t,outVals[i,:])
    plt.ylabel(str(i))
plt.show()

