"""
Demonstration of the basic operation of the SNS-Toolbox. Using scipy, the basic flow of every backend type will be demonstrated
William Nourse
May 4th, 2021
What's in the box?
"""

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
IMPORTS
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
FUNCTIONS
"""

def forward_manual(dt, Cm, Gm, Ibias, gMax, delE, R, Ulast):
    """
    Compute the network manually
    :param dt:      Simulation timestep (ms)
    :param Cm:      Vector of membrane capacitance terms (nF)
    :param Gm:      Vector of membrane conductance terms (uS)
    :param Ibias:   Vector of bias currents (nA)
    :param gMax:    Matrix of max synaptic conductances (uS)
    :param delE:    Matrix of synaptic reversal potentials (mV)
    :param R:       Range of neural activity (mV)
    :param Ulast:   Vector of voltages from previous time step (mV)

    :return Unext:  Vector of voltages for next time step (mV)
    """
    numElements = len(Cm)   # Number of nodes in the network
    Unext = np.zeros(numElements)   # Initialize empty vector to store all of the new timestep voltages
    Gsyn = np.zeros([numElements,numElements]) # Initialize empty matrix to store all synaptic conductances (this can get large)

    for source in range(numElements):
        for dest in range(numElements):
            Gsyn[dest,source] = gMax[dest,source] * max(0,min(1,Ulast[source]/R))   # Update the conductance of every synapse

    for i in range(numElements):
        Isyn = 0

        for pre in range(numElements):
            Isyn = Isyn + Gsyn[i,pre]*(delE[i,pre] - Ulast[i])  # Increment the synaptic current from each synapse

        Unext[i] = Ulast[i] + dt / Cm[i] * (-Gm[i] * Ulast[i] + Ibias[i] + Isyn)   # Update the neuron state

    return Unext


def forward_scipy(timeFactor, Gm, Ibias, gMax, delE, R, Ulast):
    """
    Compute the network states using sparse matrix techniques (scipy)
    :param Cm:      Sparse vector of dt/Cm (ms/nF)
    :param Gm:      Sparse matrix of membrane conductance terms, which is only nonzero on the diagonal (uS)
    :param Ibias:   Sparse vector of bias currents (nA)
    :param gMax:    Sparse matrix of maximum synaptic conductances (uS)
    :param delE:    Sparse matrix of synaptic reversal potentials (mV)
    :param R:       Range of neural activity (mV)
    :param Ulast:   Sparse vector of voltages from previous time step (mV)

    :return Unext:  Sparse vector of voltages for next time step
    """
    numElements = np.size(Cm)   # Number of neurons in the network

    # Want to compute Gsyn = max(0, min(gMax, gMax*Ulast/R)), using sparse operations
    # Gsyn = gMax.copy()
    # temp = gMax*Ulast/R
    # Gsyn = Gsyn.minimum(temp)
    # Gsyn = Gsyn.maximum(0)
    Gsyn = gMax*np.maximum(0,np.minimum(1,Ulast/R))

    Isyn = np.zeros(numElements)    # Initialize synaptic current for all neurons

    # Compute the following for each neuron:
    # Isyn[j] = sum(elementwise(G[:,j], delE[:,j])) - Ulast[j]*sum(G[:,j])
    for i in range(numElements):
        # temp1 = Gsyn[:,i]
        # temp2 = delE[:,i]
        # delTerm = Gsyn[:,i].multiply(delE[:,i])
        # Isyn[i,0] = delTerm.sum() - Ulast[i,0]*Gsyn[:,i].sum()
        Isyn[i] = np.sum(Gsyn[i,:]*delE[i,:]) - Ulast[i]*np.sum(Gsyn[i,:])
        foo=0

    Unext = Ulast + timeFactor*(-(Ulast @ Gm) + Ibias + Isyn)  # Update all of the neurons at once

    return Unext

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
PARAMETER DEFINITIONS
"""

dt = 0.1  # ms
numNeurons = 2
tMax = 100
R = 20
Cm = 5 + np.zeros(numNeurons)  # nF
Gm = 1 + np.zeros(numNeurons)  # uS
GmArr = np.diag(Gm) # uS, Converted into a diagonal matrix
#Ibias = np.linspace(0,R,num=numNeurons)
Ibias = np.array([R,0])
t = np.arange(0,tMax,dt)
numSteps = len(t)
Umanual = np.zeros([numNeurons,numSteps])
Umatrix = np.zeros([numSteps,numNeurons])

delE = 100
delEMat = np.array([[0,0],[delE,0]])
gMax = R/(delE-R)
gMaxMat = np.array([[0,0],[gMax,0]])

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SIMULATION
"""

# Compute the network manually
for i in range(1,numSteps):
    Umanual[:,i] = forward_manual(dt, Cm, Gm, Ibias, gMaxMat, delEMat, R, Umanual[:, i - 1])

# Compute the network with matrices/vectors
for i in range(1,numSteps):
    Umatrix[i,:] = forward_scipy(dt/Cm, GmArr, Ibias, gMaxMat, delEMat, R, Umatrix[i-1,:])

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
PLOTTING
"""

# Manual
plt.figure()
for i in range(numNeurons):
    plt.plot(t,Umanual[i,:],label=str(i))
plt.xlabel('t (ms)')
plt.ylabel('U (mV)')
plt.title('Manual Computation')
plt.legend()


# Matrix
plt.figure()
for i in range(numNeurons):
    plt.plot(t,Umatrix[:,i].transpose(),label=str(i))
plt.xlabel('t (ms)')
plt.ylabel('U (mV)')
plt.title('Matrix Computation')
plt.legend()

# Difference
plt.figure()
for i in range(numNeurons):
    plt.plot(t,Umanual[i,:]-Umatrix[:,i].transpose(),label=str(i))
plt.xlabel('t (ms)')
plt.ylabel('U (mV)')
plt.title('Manual - Matrix')
plt.legend()

plt.show()
