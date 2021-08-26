"""
Implement the Numpy backend, and collect timing information with different parameters
William Nourse
August 26th, 2021
I have set myself beyond the pale. I am nothing. I am hardly human anymore.
"""

import numpy as np
import pickle
import time
import sys

"""
########################################################################################################################
NETWORK STEP

Update all of the neural states for 1 timestep
"""

def stepAll(inputConnectivity,inputVals,Ulast,timeFactorMembrane,Gm,Ib,thetaLast,timeFactorThreshold,theta0,m,refCtr,
            refPeriod,GmaxNon,GmaxSpk,Gspike,timeFactorSynapse,DelE,outputConnectivity,R=20):
    """
    All components are present
    :param inputConnectivity:   Matrix describing routing of input currents
    :param inputVals:           Value of input currents (nA)
    :param Ulast:               Vector of neural states at the previous timestep (mV)
    :param timeFactorMembrane:  Vector of constant parameters for each neuron (dt/Cm)
    :param Gm:                  Vector of membrane conductances (uS)
    :param Ib:                  Vector of bias currents (nA)
    :param thetaLast:           Firing threshold at the previous timestep (mV)
    :param timeFactorThreshold: Vector of constant parameters for each neuron (dt/tauTheta)
    :param theta0:              Vector of initial firing thresholds (mV)
    :param m:                   Vector of threshold adaptation ratios
    :param refCtr:              Vector to store remaining timesteps in the refractory period
    :param refPeriod:           Vector of refractory periods
    :param GmaxNon:             Matrix of maximum nonspiking synaptic conductances (uS)
    :param GmaxSpk:             Matrix of maximum spiking synaptic conductances (uS)
    :param Gspike:              Matrix of spiking synaptic conductances (uS)
    :param timeFactorSynapse:   Matrix of constant parameters for each synapse (dt/tauSyn)
    :param DelE:                Matrix of synaptic reversal potentials
    :param outputConnectivity:  Matrix describing routes to output nodes
    :param R:                   Neural range (mV)

    :return: U, Ulast, thetaLast, Gspike, refCtr, outputNodes
    """
    Iapp = np.matmul(inputConnectivity,inputVals)  # Apply external current sources to their destinations
    Gnon = np.maximum(0, np.minimum(GmaxNon * Ulast/R, GmaxNon))
    Gspike = Gspike * (1 - timeFactorSynapse)
    Gsyn = Gnon + Gspike
    Isyn = np.sum(Gsyn * DelE, axis=1) - Ulast * np.sum(Gsyn, axis=1)
    U = Ulast + timeFactorMembrane * (-Gm * Ulast + Ib + Isyn + Iapp)  # Update membrane potential
    theta = thetaLast + timeFactorThreshold * (-thetaLast + theta0 + m * Ulast)  # Update the firing thresholds
    spikes = np.sign(np.minimum(0, theta + U * (-1 + refCtr)))  # Compute which neurons have spiked
    Gspike = np.maximum(Gspike, (-spikes) * GmaxSpk)  # Update the conductance of synapses which spiked
    U = U * (spikes + 1)  # Reset the membrane voltages of neurons which spiked
    refCtr = np.maximum(0, refCtr - spikes * (refPeriod + 1) - 1)  # Update refractory periods
    outputNodes = np.matmul(U, outputConnectivity)  # Copy desired neural quantities to output nodes
    Ulast = np.copy(U)  # Copy the current membrane voltage to be the past value
    thetaLast = np.copy(theta)  # Copy the current threshold value to be the past value

    return U, Ulast, thetaLast, Gspike, refCtr, outputNodes

def stepNoRef(inputConnectivity,inputVals,Ulast,timeFactorMembrane,Gm,Ib,thetaLast,timeFactorThreshold,theta0,m,GmaxNon,
              GmaxSpk,Gspike,timeFactorSynapse,DelE,outputConnectivity,R=20):
    """
    There is no refractory period
    :param inputConnectivity:   Matrix describing routing of input currents
    :param inputVals:           Value of input currents (nA)
    :param Ulast:               Vector of neural states at the previous timestep (mV)
    :param timeFactorMembrane:  Vector of constant parameters for each neuron (dt/Cm)
    :param Gm:                  Vector of membrane conductances (uS)
    :param Ib:                  Vector of bias currents (nA)
    :param thetaLast:           Firing threshold at the previous timestep (mV)
    :param timeFactorThreshold: Vector of constant parameters for each neuron (dt/tauTheta)
    :param theta0:              Vector of initial firing thresholds (mV)
    :param m:                   Vector of threshold adaptation ratios
    :param GmaxNon:             Matrix of maximum nonspiking synaptic conductances (uS)
    :param GmaxSpk:             Matrix of maximum spiking synaptic conductances (uS)
    :param Gspike:              Matrix of spiking synaptic conductances (uS)
    :param timeFactorSynapse:   Matrix of constant parameters for each synapse (dt/tauSyn)
    :param DelE:                Matrix of synaptic reversal potentials
    :param outputConnectivity:  Matrix describing routes to output nodes
    :param R:                   Range of neural activity (mV)

    :return: U, Ulast, thetaLast, Gspike, outputNodes
    """
    Iapp = np.matmul(inputConnectivity,inputVals)  # Apply external current sources to their destinations
    Gnon = np.maximum(0, np.minimum(GmaxNon * Ulast/R, GmaxNon))
    Gspike = Gspike * (1 - timeFactorSynapse)
    Gsyn = Gnon + Gspike
    Isyn = np.sum(Gsyn * DelE, axis=1) - Ulast * np.sum(Gsyn, axis=1)
    U = Ulast + timeFactorMembrane * (-Gm * Ulast + Ib + Isyn + Iapp)  # Update membrane potential
    theta = thetaLast + timeFactorThreshold * (-thetaLast + theta0 + m * Ulast)  # Update the firing thresholds
    spikes = np.sign(np.minimum(0, theta - U))  # Compute which neurons have spiked
    Gspike = np.maximum(Gspike, (-spikes) * GmaxSpk)  # Update the conductance of synapses which spiked
    U = U * (spikes + 1)  # Reset the membrane voltages of neurons which spiked
    outputNodes = np.matmul(U, outputConnectivity)  # Copy desired neural quantities to output nodes
    Ulast = np.copy(U)  # Copy the current membrane voltage to be the past value
    thetaLast = np.copy(theta)  # Copy the current threshold value to be the past value

    return U, Ulast, thetaLast, Gspike, outputNodes

def stepNoSpike(inputConnectivity,inputVals,Ulast,timeFactorMembrane,Gm,Ib,GmaxNon,DelE,outputConnectivity,R=20):
    """
    No neurons can be spiking
    :param inputConnectivity:   Matrix describing routing of input currents
    :param inputVals:           Value of input currents (nA)
    :param Ulast:               Vector of neural states at the previous timestep (mV)
    :param timeFactorMembrane:  Vector of constant parameters for each neuron (dt/Cm)
    :param Gm:                  Vector of membrane conductances (uS)
    :param Ib:                  Vector of bias currents (nA)
    :param GmaxNon:             Matrix of maximum nonspiking synaptic conductances (uS)
    :param DelE:                Matrix of synaptic reversal potentials
    :param outputConnectivity:  Matrix describing routes to output nodes
    :param R:                   Range of neural activity (mV)

    :return: U, Ulast, outputNodes
    """
    Iapp = np.matmul(inputConnectivity,inputVals)  # Apply external current sources to their destinations
    Gsyn = np.maximum(0, np.minimum(GmaxNon * Ulast/R, GmaxNon))
    Isyn = np.sum(Gsyn * DelE, axis=1) - Ulast * np.sum(Gsyn, axis=1)
    U = Ulast + timeFactorMembrane * (-Gm * Ulast + Ib + Isyn + Iapp)  # Update membrane potential
    outputNodes = np.matmul(U, outputConnectivity)  # Copy desired neural quantities to output nodes
    Ulast = np.copy(U)  # Copy the current membrane voltage to be the past value

    return U, Ulast, outputNodes

"""
########################################################################################################################
NETWORK CONSTRUCTION

Construct testing networks using specifications
"""

def constructAll(dt,numNeurons,perConn,perIn,perOut,perSpike,seed=0):
    """
    All elements are present
    :param dt:          Simulation timestep (ms)
    :param numNeurons:  Number of neurons in the network
    :param perConn:     Percent of network which is connected
    :param perIn:       Percent of input nodes in the network
    :param perOut:      Percent of output nodes in the network
    :param perSpike:    Percent of neurons which are spiking
    :param seed:        Random seed
    :return:            All of the parameters required to run a network
    """

    # Inputs
    numInputs = int(perIn*numNeurons)
    inputVals = np.zeros(numInputs)+1.0
    inputConnectivity = np.zeros([numNeurons,numInputs]) + 1

    # Construct neurons
    Ulast = np.zeros(numNeurons)
    numSpike = int(perSpike*numNeurons)
    Cm = np.zeros(numNeurons) + 5.0  # membrane capacitance (nF)
    Gm = np.zeros(numNeurons) + 1.0  # membrane conductance (uS)
    Ib = np.zeros(numNeurons) + 10.0  # bias current (nA)
    timeFactorMembrane = dt/Cm

    # Threshold stuff
    theta0 = np.zeros(numNeurons)
    for i in range(numNeurons):
        if i >= numSpike:
            theta0[i] = sys.float_info.max
        else:
            theta0[i] = 1.0
    thetaLast = np.copy(theta0)
    m = np.zeros(numNeurons)
    tauTheta = np.zeros(numNeurons)+1.0
    timeFactorThreshold = dt/tauTheta

    # Refractory period
    refCtr = np.zeros(numNeurons)
    refPeriod = np.zeros(numNeurons)+1

    # Synapses
    GmaxNon = np.zeros([numNeurons,numNeurons])
    GmaxSpk = np.zeros([numNeurons,numNeurons])
    Gspike = np.zeros([numNeurons,numNeurons])
    DelE = np.zeros([numNeurons,numNeurons])
    tauSyn = np.zeros([numNeurons, numNeurons])+1
    numSyn = int(perConn*numNeurons*numNeurons)
    k = 0
    usedIndex = []
    np.random.seed(seed)
    while k<numSyn:
        row = np.random.randint(0,numNeurons-1)
        col = np.random.randint(0, numNeurons - 1)
        while [row,col] in usedIndex:
            row = np.random.randint(0, numNeurons - 1)
            col = np.random.randint(0, numNeurons - 1)
        usedIndex.append([row,col])
        k += 1

        DelE[row][col] = 100
        if theta0[col] < sys.float_info.max:
            GmaxSpk[row][col] = 1
        else:
            GmaxNon[row][col] = 1
        tauSyn[row][col] = 2

    timeFactorSynapse = dt/tauSyn

    # Outputs
    numOutputs = int(perOut*numNeurons)
    outputConnectivity = np.zeros([numOutputs,numNeurons])
    for i in range(numOutputs):
        outputConnectivity[i][i] = 1

    return (inputConnectivity,inputVals,Ulast,timeFactorMembrane,Gm,Ib,thetaLast,timeFactorThreshold,theta0,m,refCtr,
            refPeriod,GmaxNon,GmaxSpk,Gspike,timeFactorSynapse,DelE,outputConnectivity)

def constructNoRef(dt,numNeurons,perConn,perIn,perOut,perSpike,seed=0):
    """
    No refractory period
    :param dt:          Simulation timestep (ms)
    :param numNeurons:  Number of neurons in the network
    :param perConn:     Percent of network which is connected
    :param perIn:       Percent of input nodes in the network
    :param perOut:      Percent of output nodes in the network
    :param perSpike:    Percent of neurons which are spiking
    :param seed:        Random seed
    :return:            All of the parameters required to run a network
    """

    # Inputs
    numInputs = int(perIn*numNeurons)
    inputVals = np.zeros(numInputs)+1.0
    inputConnectivity = np.zeros([numNeurons,numInputs]) + 1

    # Construct neurons
    Ulast = np.zeros(numNeurons)
    numSpike = int(perSpike*numNeurons)
    Cm = np.zeros(numNeurons) + 5.0  # membrane capacitance (nF)
    Gm = np.zeros(numNeurons) + 1.0  # membrane conductance (uS)
    Ib = np.zeros(numNeurons) + 10.0  # bias current (nA)
    timeFactorMembrane = dt/Cm

    # Threshold stuff
    theta0 = np.zeros(numNeurons)
    for i in range(numNeurons):
        if i >= numSpike:
            theta0[i] = sys.float_info.max
        else:
            theta0[i] = 1.0
    thetaLast = np.copy(theta0)
    m = np.zeros(numNeurons)
    tauTheta = np.zeros(numNeurons)+1.0
    timeFactorThreshold = dt/tauTheta

    # Synapses
    GmaxNon = np.zeros([numNeurons,numNeurons])
    GmaxSpk = np.zeros([numNeurons,numNeurons])
    Gspike = np.zeros([numNeurons,numNeurons])
    DelE = np.zeros([numNeurons,numNeurons])
    tauSyn = np.zeros([numNeurons, numNeurons])
    numSyn = int(perConn*numNeurons*numNeurons)
    k = 0
    usedIndex = []
    np.random.seed(seed)
    while k<numSyn:
        row = np.random.randint(0, numNeurons - 1)
        col = np.random.randint(0, numNeurons - 1)
        while [row, col] in usedIndex:
            row = np.random.randint(0, numNeurons - 1)
            col = np.random.randint(0, numNeurons - 1)
        usedIndex.append([row, col])
        k += 1

        DelE[row][col] = 100
        if theta0[col] < sys.float_info.max:
            GmaxSpk[row][col] = 1
        else:
            GmaxNon[row][col] = 1
        tauSyn[row][col] = 1
    timeFactorSynapse = dt/tauSyn

    # Outputs
    numOutputs = int(perOut*numNeurons)
    outputConnectivity = np.zeros([numOutputs,numNeurons])
    for i in range(numOutputs):
        outputConnectivity[i][i] = 1

    return (inputConnectivity,inputVals,Ulast,timeFactorMembrane,Gm,Ib,thetaLast,timeFactorThreshold,theta0,m,GmaxNon,
            GmaxSpk,Gspike,timeFactorSynapse,DelE,outputConnectivity)

def constructNoSpike(dt,numNeurons,perConn,perIn,perOut,seed=0):
    """
    No spiking elements
    :param dt:          Simulation timestep (ms)
    :param numNeurons:  Number of neurons in the network
    :param perConn:     Percent of network which is connected
    :param perIn:       Percent of input nodes in the network
    :param perOut:      Percent of output nodes in the network
    :param seed:        Random seed
    :return:            All of the parameters required to run a network
    """

    # Inputs
    numInputs = int(perIn*numNeurons)
    inputVals = np.zeros(numInputs)+1.0
    inputConnectivity = np.zeros([numNeurons,numInputs]) + 1

    # Construct neurons
    Ulast = np.zeros(numNeurons)
    Cm = np.zeros(numNeurons) + 5.0  # membrane capacitance (nF)
    Gm = np.zeros(numNeurons) + 1.0  # membrane conductance (uS)
    Ib = np.zeros(numNeurons) + 10.0  # bias current (nA)
    timeFactorMembrane = dt/Cm

    # Synapses
    GmaxNon = np.zeros([numNeurons,numNeurons])
    DelE = np.zeros([numNeurons,numNeurons])
    numSyn = int(perConn*numNeurons*numNeurons)
    k = 0
    usedIndex = []
    np.random.seed(seed)
    while k<numSyn:
        row = np.random.randint(0, numNeurons - 1)
        col = np.random.randint(0, numNeurons - 1)
        while [row, col] in usedIndex:
            row = np.random.randint(0, numNeurons - 1)
            col = np.random.randint(0, numNeurons - 1)
        usedIndex.append([row, col])
        k += 1

        DelE[row][col] = 100
        GmaxNon[row][col] = 1

    # Outputs
    numOutputs = int(perOut*numNeurons)
    outputConnectivity = np.zeros([numOutputs,numNeurons])
    for i in range(numOutputs):
        outputConnectivity[i][i] = 1

    return inputConnectivity,inputVals,Ulast,timeFactorMembrane,Gm,Ib,GmaxNon,DelE,outputConnectivity

"""
########################################################################################################################
TESTING
"""
outs = constructAll(dt=0.001,numNeurons=5,perConn=0.5,perIn=0.2,perOut=0.4,perSpike=0.5)
print('Input Connectivity:')
print(outs[0])
print('\nInput Values:')
print(outs[1])
print('\nUlast:')
print(outs[2])
print('\ntimeFactorMembrane:')
print(outs[3])
print('\nGm:')
print(outs[4])
print('\nIb:')
print(outs[5])
print('\nThetaLast:')
print(outs[6])
print('\nTimeFactorThreshold:')
print(outs[7])
print('\nTheta0:')
print(outs[8])
print('\nm:')
print(outs[9])
print('\nRefCtr:')
print(outs[10])
print('\nRefPeriod:')
print(outs[11])
print('\nGmaxNon:')
print(outs[12])
print('\nGmaxSpike:')
print(outs[13])
print('\nGspike:')
print(outs[14])
print('\nTimeFactorSynapse:')
print(outs[15])
print('\nDelE:')
print(outs[16])
print('\nOutputConnectivity:')
print(outs[17])