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
import sys

"""
########################################################################################################################
NETWORK CONSTRUCTION
"""

# Network Parameters
R = 20.0    # neural activity range (mV)
numNeurons = 11 # number of neurons in the demo network
tauTheta = 1
dt = 0.001  # timestep (ms)

# External Input Handling
externalCurrent = 20.0         # magnitude of applied current (nA)
inputNodes = [externalCurrent] # vector of input currents
inputConnectivityVector = np.array([[1,1,1,1,1,1,0,0,0,0,0]])   # this is a vector because input is 1d

# Initialize Neurons
Uini = np.zeros(numNeurons)     # all neurons should start with zero voltage
theta0 = np.array([[1.0,1.0,1.0,sys.float_info.max,sys.float_info.max,1.0,1.0,sys.float_info.max,sys.float_info.max,sys.float_info.max,1.0]])
Ulast = np.copy(Uini)   # previous timestep
U = np.zeros(numNeurons)        # current timestep
thetaLast = np.copy(theta0)
theta = np.zeros([1,numNeurons])
spikes = np.zeros(numNeurons)

Gspike = np.zeros([numNeurons,numNeurons])
#                            Source
#                    0 1 2 3 4 5 6 7 8 9 10
GmaxNon = np.array([[0,0,0,0,0,0,0,0,0,0,0], # 0  D
                    [0,0,0,0,0,0,0,0,0,0,0], # 1  e
                    [0,0,0,0,0,0,0,0,0,0,0], # 2  s
                    [0,0,0,0,0,0,0,0,0,0,0], # 3  t
                    [0,0,0,0,0,0,0,0,0,0,0], # 4  i
                    [0,0,0,0,0,0,0,0,0,0,0], # 5  n
                    [0,0,0,0,0,0,0,0,0,0,0], # 6  a
                    [0,0,0,1,0,0,0,0,0,0,0], # 7  t
                    [0,0,0,1,0,0,0,0,0,0,0], # 8  i
                    [0,0,0,1,10,0,0,0,0,0,0], # 9  o
                    [0,0,0,0,10,0,0,0,0,0,0]])# 10 n

#                            Source
#                    0 1 2 3 4 5 6 7 8 9 10
GmaxSpk = np.array([[0,0,0,0,0,0,0,0,0,0,0], # 0  D
                    [0,0,0,0,0,0,0,0,0,0,0], # 1  e
                    [0,0,0,0,0,0,0,0,0,0,0], # 2  s
                    [0,0,0,0,0,0,0,0,0,0,0], # 3  t
                    [0,0,0,0,0,0,0,0,0,0,0], # 4  i
                    [1,0,0,0,0,0,0,0,0,0,0], # 5  n
                    [1,0,0,0,0,0,0,0,0,0,0], # 6  a
                    [0,0,0,0,0,0,0,0,0,0,0], # 7  t
                    [0,0,0,0,0,0,0,0,0,0,0], # 8  i
                    [0,0,0,0,0,0,0,0,0,0,0], # 9  o
                    [1,0,0,0,0,0,0,0,0,0,0]])# 10 n

#                            Source
#                    0 1 2 3 4 5 6 7 8 9 10
DelE = np.array([[0,0,0,0,0,0,0,0,0,0,0],    # 0  D
                 [0,0,0,0,0,0,0,0,0,0,0],    # 1  e
                 [0,0,0,0,0,0,0,0,0,0,0],    # 2  s
                 [0,0,0,0,0,0,0,0,0,0,0],    # 3  t
                 [0,0,0,0,0,0,0,0,0,0,0],    # 4  i
                 [-100,0,0,0,0,0,0,0,0,0,0],    # 5  n
                 [100,0,0,0,0,0,0,0,0,0,0],    # 6  a
                 [0,0,0,-100,0,0,0,0,0,0,0], # 7  t
                 [0,0,0,100,0,0,0,0,0,0,0],  # 8  i
                 [0,0,0,100,0,0,0,0,0,0,0],  # 9  o
                 [100,0,0,0,0,0,0,0,0,0,0]])   # 10 n

# Neural Parameters
Cm = np.zeros(numNeurons) + 5.0 # membrane capacitance (nF)
Gm = np.zeros(numNeurons) + 1.0 # membrane conductance (uS)
Ib = np.zeros(numNeurons) + 0.0 # bias current (nA)
m = np.array([0.0,-1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])

# Output Nodes
outputNodes = np.zeros(numNeurons)  # we will read the voltage from every neuron in the network
outputConnectivityMatrix = np.identity(numNeurons)  # matrix because we're reading every neural voltage

"""
########################################################################################################################
SIMULATION
"""
tmax = 50                 # max time (ms)
t = np.arange(0,tmax,dt)    # simulation time vector
numSteps = len(t)           # number of simulation steps
timeFactorMembrane = dt / Cm          # multiplicative time factor (save on operations)
timeFactorThreshold = dt / tauTheta
tauSyn = 1
timeFactorSynapse = dt / tauSyn

outVals = np.zeros([numNeurons,numSteps])  # vector for storing output values

refCtr = np.zeros(numNeurons)
refMs = 0 # refractory period in ms
refPeriod = np.zeros(numNeurons)+refMs*(1/dt)

for i in range(numSteps):
    Iapp = inputConnectivityVector*inputNodes   # Apply external current sources to their destinations
    Gnon = np.maximum(0,np.minimum(GmaxNon*Ulast,GmaxNon))
    Gspike = Gspike*(1 - timeFactorSynapse)
    Gsyn = Gnon + Gspike
    Isyn = np.sum(Gsyn*DelE,axis=1) - Ulast*np.sum(Gsyn,axis=1)
    U = Ulast + timeFactorMembrane * (-Gm * Ulast + Ib + Isyn + Iapp)  # Update membrane potential
    theta = thetaLast + timeFactorThreshold * (-thetaLast+theta0+m*Ulast)   # Update the firing thresholds
    spikes = np.sign(np.minimum(0, theta + U*(-1 + refCtr)))    # Compute which neurons have spiked
    Gspike = np.maximum(Gspike, (-spikes) * GmaxSpk)    # Update the conductance of synapses which spiked
    U = U*(spikes+1)    # Reset the membrane voltages of neurons which spiked
    refCtr = np.maximum(0, refCtr - spikes * (refPeriod + 1) - 1) # Update refractory periods
    outputNodes = np.matmul(U,outputConnectivityMatrix) # Copy desired neural quantities to output nodes
    outVals[:,i] = outputNodes  # Read output values

    Ulast = np.copy(U)  # Copy the current membrane voltage to be the past value
    thetaLast = np.copy(theta)  # Copy the current threshold value to be the past value

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

