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

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
FUNCTIONS
"""

def forward_manual(dt, Cm, Gm, Ibias, Ulast):
    """
    Compute the network manually
    :param dt:      Simulation timestep
    :param Cm:      Vector of membrane capacitance terms
    :param Gm:      Vector of membrane conductance terms
    :param Ibias:   Vector of bias currents
    :param Ulast:   Vector of voltages from previous time step

    :return Unext:  Vector of voltages for next time step
    """
    numElements = len(Cm)   # Number of nodes in the network
    Unext = np.zeros(numElements)   # Initialize empty vector to store all of the new timestep voltages
    for i in range(numElements):    # Iterate over every neuron
        Unext[i] = Ulast[i] + dt / Cm[i] * (-Gm[i] * Ulast[i] + Ibias[i])   # Update the neuron state

    return Unext


def forward_matrix(dt, Cm, Gm, Ibias, Ulast):
    """
    Compute the network states using matrix techniques
    :param dt:      Simulation timestep
    :param Cm:      Vector of membrane capacitance terms
    :param Gm:      Matrix of membrane conductance terms, which is only nonzero on the diagonal
    :param Ibias:   Vector of bias currents
    :param Ulast:   Vector of voltages from previous time step

    :return Unext:  Vector of voltages for next time step
    """
    Unext = Ulast + dt / Cm * (-Gm.dot(Ulast) + Ibias)  # Update all of the neurons at once

    return Unext

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
PARAMETER DEFINITIONS
"""

dt = 1  # ms
numNeurons = 5
tMax = 100
R = 20
Cm = 5 + np.zeros(numNeurons)  # nF
Gm = 1 + np.zeros(numNeurons)  # uS
GmArr = np.diag(Gm) # uS, Converted into a diagonal matrix
Ibias = np.linspace(0,R,num=numNeurons)
t = np.arange(0,tMax,dt)
numSteps = len(t)
Umanual = np.zeros([numNeurons,numSteps])
Umatrix = np.zeros([numNeurons,numSteps])

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SIMULATION
"""

# Compute the network manually
for i in range(1,numSteps):
    Umanual[:,i] = forward_manual(dt,Cm,Gm,Ibias,Umanual[:,i-1])

# Compute the network with matrices/vectors
for i in range(1,numSteps):
    Umatrix[:,i] = forward_matrix(dt,Cm,GmArr,Ibias,Umatrix[:,i-1])

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
    plt.plot(t,Umatrix[i,:],label=str(i))
plt.xlabel('t (ms)')
plt.ylabel('U (mV)')
plt.title('Matrix Computation')
plt.legend()

# Difference
plt.figure()
for i in range(numNeurons):
    plt.plot(t,Umanual[i,:]-Umatrix[i,:],label=str(i))
plt.xlabel('t (ms)')
plt.ylabel('U (mV)')
plt.title('Manual - Matrix')
plt.legend()

plt.show()
