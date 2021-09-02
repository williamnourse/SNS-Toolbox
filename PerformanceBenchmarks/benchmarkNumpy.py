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

def stepAll(inputConnectivity, inputVals, Ulast, timeFactorMembrane, Gm, Ib, thetaLast, timeFactorThreshold, theta0, m, refCtr,
            refPeriod, GmaxNon, GmaxSpk, Gspike, timeFactorSynapse, DelE, outputVoltageConnectivity,
            outputSpikeConnectivity, R=20):
    """
    All components are present
    :param inputConnectivity:           Matrix describing routing of input currents
    :param inputVals:                   Value of input currents (nA)
    :param Ulast:                       Vector of neural states at the previous timestep (mV)
    :param timeFactorMembrane:          Vector of constant parameters for each neuron (dt/Cm)
    :param Gm:                          Vector of membrane conductances (uS)
    :param Ib:                          Vector of bias currents (nA)
    :param thetaLast:                   Firing threshold at the previous timestep (mV)
    :param timeFactorThreshold:         Vector of constant parameters for each neuron (dt/tauTheta)
    :param theta0:                      Vector of initial firing thresholds (mV)
    :param m:                           Vector of threshold adaptation ratios
    :param refCtr:                      Vector to store remaining timesteps in the refractory period
    :param refPeriod:                   Vector of refractory periods
    :param GmaxNon:                     Matrix of maximum nonspiking synaptic conductances (uS)
    :param GmaxSpk:                     Matrix of maximum spiking synaptic conductances (uS)
    :param Gspike:                      Matrix of spiking synaptic conductances (uS)
    :param timeFactorSynapse:           Matrix of constant parameters for each synapse (dt/tauSyn)
    :param DelE:                        Matrix of synaptic reversal potentials
    :param outputVoltageConnectivity:   Matrix describing routes to output nodes
    :param outputSpikeConnectivity:     Matrix describing routes to output nodes
    :param R:                           Neural range (mV)

    :return: U, Ulast, thetaLast, Gspike, refCtr, outputVoltages
    """
    start = time.time()

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
    outputVoltages = np.matmul(outputVoltageConnectivity, U)  # Copy desired neural quantities to output nodes
    outputSpikes = np.matmul(outputSpikeConnectivity, spikes)  # Copy desired neural quantities to output nodes
    Ulast = np.copy(U)  # Copy the current membrane voltage to be the past value
    thetaLast = np.copy(theta)  # Copy the current threshold value to be the past value

    end = time.time()
    return U, Ulast, thetaLast, Gspike, refCtr, outputVoltages, outputSpikes, end-start

def stepNoRef(inputConnectivity, inputVals, Ulast, timeFactorMembrane, Gm, Ib, thetaLast, timeFactorThreshold, theta0, m, GmaxNon,
              GmaxSpk, Gspike, timeFactorSynapse, DelE, outputVoltageConnectivity, outputSpikeConnectivity, R=20):
    """
    There is no refractory period
    :param inputConnectivity:           Matrix describing routing of input currents
    :param inputVals:                   Value of input currents (nA)
    :param Ulast:                       Vector of neural states at the previous timestep (mV)
    :param timeFactorMembrane:          Vector of constant parameters for each neuron (dt/Cm)
    :param Gm:                          Vector of membrane conductances (uS)
    :param Ib:                          Vector of bias currents (nA)
    :param thetaLast:                   Firing threshold at the previous timestep (mV)
    :param timeFactorThreshold:         Vector of constant parameters for each neuron (dt/tauTheta)
    :param theta0:                      Vector of initial firing thresholds (mV)
    :param m:                           Vector of threshold adaptation ratios
    :param GmaxNon:                     Matrix of maximum nonspiking synaptic conductances (uS)
    :param GmaxSpk:                     Matrix of maximum spiking synaptic conductances (uS)
    :param Gspike:                      Matrix of spiking synaptic conductances (uS)
    :param timeFactorSynapse:           Matrix of constant parameters for each synapse (dt/tauSyn)
    :param DelE:                        Matrix of synaptic reversal potentials
    :param outputVoltageConnectivity:   Matrix describing routes to output nodes
    :param outputSpikeConnectivity:     Matrix describing routes to output nodes
    :param R:                           Range of neural activity (mV)

    :return: U, Ulast, thetaLast, Gspike, outputVoltages, outputSpikes
    """
    start = time.time()

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
    outputVoltages = np.matmul(outputVoltageConnectivity, U)  # Copy desired neural quantities to output nodes
    outputSpikes = np.matmul(outputSpikeConnectivity, spikes)  # Copy desired neural quantities to output nodes
    Ulast = np.copy(U)  # Copy the current membrane voltage to be the past value
    thetaLast = np.copy(theta)  # Copy the current threshold value to be the past value

    end = time.time()
    return U, Ulast, thetaLast, Gspike, outputVoltages, outputSpikes, end - start

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
    start = time.time()

    Iapp = np.matmul(inputConnectivity,inputVals)  # Apply external current sources to their destinations
    Gsyn = np.maximum(0, np.minimum(GmaxNon * Ulast/R, GmaxNon))
    Isyn = np.sum(Gsyn * DelE, axis=1) - Ulast * np.sum(Gsyn, axis=1)
    U = Ulast + timeFactorMembrane * (-Gm * Ulast + Ib + Isyn + Iapp)  # Update membrane potential
    outputNodes = np.matmul(outputConnectivity,U)  # Copy desired neural quantities to output nodes
    Ulast = np.copy(U)  # Copy the current membrane voltage to be the past value

    end = time.time()
    return U, Ulast, outputNodes,end-start

"""
########################################################################################################################
NETWORK CONSTRUCTION

Construct testing networks using specifications
"""

def constructAll(dt, numNeurons, probConn, perIn, perOut, perSpike, seed=0):
    """
    All elements are present
    :param dt:          Simulation timestep (ms)
    :param numNeurons:  Number of neurons in the network
    :param probConn:     Percent of network which is connected
    :param perIn:       Percent of input nodes in the network
    :param perOut:      Percent of output nodes in the network
    :param perSpike:    Percent of neurons which are spiking
    :param seed:        Random seed
    :return:            All of the parameters required to run a network
    """

    # Inputs
    numInputs = int(perIn*numNeurons)
    if numInputs == 0:
        numInputs = 1
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

    np.random.seed(seed)
    for row in range(numNeurons):
        for col in range(numNeurons):
            rand = np.random.uniform()
            if rand < probConn:
                DelE[row][col] = 100
                if theta0[col] < sys.float_info.max:
                    GmaxSpk[row][col] = 1
                else:
                    GmaxNon[row][col] = 1
                tauSyn[row][col] = 2

    timeFactorSynapse = dt/tauSyn

    # Outputs
    numOutputs = int(perOut*numNeurons)
    if numOutputs == 0:
        numOutputs = 1
    outputVoltageConnectivity = np.zeros([numOutputs,numNeurons])
    for i in range(numOutputs):
        outputConnectivity[i][i] = 1
    outputSpikeConnectivity = np.copy(outputVoltageConnectivity)

    return (inputConnectivity,inputVals,Ulast,timeFactorMembrane,Gm,Ib,thetaLast,timeFactorThreshold,theta0,m,refCtr,
            refPeriod,GmaxNon,GmaxSpk,Gspike,timeFactorSynapse,DelE,outputVoltageConnectivity,outputSpikeConnectivity)

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
    tauSyn = np.zeros([numNeurons, numNeurons])+1
    numSyn = int(perConn*numNeurons*numNeurons)
    np.random.seed(seed)
    for row in range(numNeurons):
        for col in range(numNeurons):
            rand = np.random.uniform()
            if rand < probConn:
                DelE[row][col] = 100
                if theta0[col] < sys.float_info.max:
                    GmaxSpk[row][col] = 1
                else:
                    GmaxNon[row][col] = 1
                tauSyn[row][col] = 2
    timeFactorSynapse = dt/tauSyn

    # Outputs
    numOutputs = int(perOut*numNeurons)
    outputVoltageConnectivity = np.zeros([numOutputs, numNeurons])
    for i in range(numOutputs):
        outputConnectivity[i][i] = 1
    outputSpikeConnectivity = np.copy(outputVoltageConnectivity)

    return (inputConnectivity, inputVals, Ulast, timeFactorMembrane, Gm, Ib, thetaLast, timeFactorThreshold, theta0, m,
            GmaxNon, GmaxSpk, Gspike, timeFactorSynapse, DelE, outputVoltageConnectivity, outputSpikeConnectivity)

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
    for row in range(numNeurons):
        for col in range(numNeurons):
            rand = np.random.uniform()
            if rand < probConn:
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
# All components:

# Testing parameters
dt = 0.001
perIn = 0.08
perOut = 0.12
numSizeSamples = 100
numSpikeSamples = 1
numConnSamples = 10
numSteps = 1000
networkSize = np.logspace(1,4,num=numSizeSamples)
# percentSpiking = np.linspace(0.0,1.0,num=numSpikeSamples)
percentSpiking = [0]
probConnectivity = np.logspace(0, 1, num=numConnSamples) / 10
start = time.time()

# parameters = {'networkSize': networkSize,
#               'probConnectivity': probConnectivity}
#
# # Testing data (no spike)
# timeData = np.zeros([numSizeSamples,numConnSamples,numSteps])
# data = {'dim1': 'networkSize',
#         'dim2': 'probConnectivity'}

# # Collection loop (no spike)
# for size in range(numSizeSamples):
#     for probConn in range(numConnSamples):
#         print('No Spike: Size %d/%d, Percent Spiking 0/0, Percent Connectivity %d/%d' % ((size+1), numSizeSamples, (probConn + 1), numConnSamples))
#         print('Running for %f seconds' % (time.time() - start))
#         (inputConnectivity,inputVals,Ulast,timeFactorMembrane,Gm,Ib,
#          GmaxNon,DelE,outputConnectivity) = constructNoSpike(dt, int(networkSize[size]),probConnectivity[probConn],perIn,perOut)
#         tStep = np.zeros(numSteps)
#         for step in range(numSteps):
#             # print('     %d'%step)
#             (_,Ulast,_,tStep[step]) = stepNoSpike(inputConnectivity,inputVals,Ulast,
#                                                   timeFactorMembrane,Gm,Ib,GmaxNon,DelE,outputConnectivity)
#         timeData[size][probConn][:] = tStep
#
# data['data'] = timeData
# numpyNoSpikeTest = {'params': parameters,'data': data}
# pickle.dump(numpyNoSpikeTest, open('dataNumpyNoSpike.p','wb'))

parameters = {'networkSize': networkSize,
              'percentSpiking': percentSpiking,
              'probConnectivity': probConnectivity}

# Testing data (no ref)
timeData = np.zeros([numSizeSamples,numSpikeSamples,numConnSamples,numSteps])
data = {'dim1': 'networkSize',
        'dim2': 'percentSpiking',
        'dim3': 'probConnectivity'}

# Collection loop (no ref)
for size in range(numSizeSamples):
    for perSpike in range(numSpikeSamples):
        for probConn in range(numConnSamples):
            print('No Ref: Size %d/%d, Percent Spiking %d/%d, Percent Connectivity %d/%d' % ((size+1), numSizeSamples, (perSpike+1), numSpikeSamples, (probConn + 1), numSizeSamples))
            print('Running for %f seconds' % (time.time() - start))
            (inputConnectivity,inputVals,Ulast,timeFactorMembrane,Gm,Ib,thetaLast, timeFactorThreshold, theta0, m,
             GmaxNon,GmaxSpk,
             Gspike,timeFactorSynapse,DelE,
             outputVoltageConnectivity,outputSpikeConnectivity) = constructNoRef(dt, int(networkSize[size]),
                                                                                 probConnectivity[probConn], perIn,
                                                                                 perOut,percentSpiking[perSpike])
            tStep = np.zeros(numSteps)
            for step in range(numSteps):
                # print('     %d'%step)
                (_,Ulast,thetaLast,Gspike,_,_,tStep[step]) = stepNoRef(inputConnectivity,inputVals,Ulast,
                                                                     timeFactorMembrane,Gm,Ib,thetaLast,
                                                                     timeFactorThreshold,theta0,m,GmaxNon,GmaxSpk,
                                                                     Gspike,timeFactorSynapse,DelE,outputVoltageConnectivity,outputSpikeConnectivity)
            timeData[size][perSpike][probConn][:] = tStep

data['data'] = timeData
numpyNoRefTest = {'params': parameters,'data': data}
pickle.dump(numpyNoRefTest, open('dataNumpyNoRef.p','wb'))

parameters = {'networkSize': networkSize,
              'percentSpiking': percentSpiking,
              'probConnectivity': probConnectivity}

# Testing data (all)
timeData = np.zeros([numSizeSamples,numSpikeSamples,numConnSamples,numSteps])
data = {'dim1': 'networkSize',
        'dim2': 'percentSpiking',
        'dim3': 'probConnectivity'}

# Collection loop (all)
for size in range(numSizeSamples):
    for perSpike in range(numSpikeSamples):
        for probConn in range(numConnSamples):
            print('All: Size %d/%d, Percent Spiking %d/%d, Percent Connectivity %d/%d' % ((size+1), numSizeSamples, (perSpike+1), numSpikeSamples, (probConn + 1), numConnSamples))
            print('Running for %f seconds'%(time.time()-start))
            (inputConnectivity,inputVals,Ulast,timeFactorMembrane,Gm,Ib,thetaLast, timeFactorThreshold, theta0, m,
             refCtr,refPeriod,GmaxNon,GmaxSpk,
             Gspike,timeFactorSynapse,DelE,outputVoltageConnectivity,outputSpikeConnectivity) = constructAll(dt, int(networkSize[size]),
                                                                                                             probConnectivity[probConn], perIn, perOut,
                                                                                                             percentSpiking[perSpike])
            tStep = np.zeros(numSteps)
            for step in range(numSteps):
                # print('     %d'%step)
                (_,Ulast,thetaLast,Gspike,refCtr,_,_,tStep[step]) = stepAll(inputConnectivity,inputVals,Ulast,
                                                                            timeFactorMembrane,Gm,Ib,thetaLast,
                                                                            timeFactorThreshold,theta0,m,refCtr,
                                                                            refPeriod,GmaxNon,GmaxSpk,Gspike,
                                                                            timeFactorSynapse,DelE,
                                                                            outputVoltageConnectivity,outputSpikeConnectivity)
            timeData[size][perSpike][probConn][:] = tStep

data['data'] = timeData
numpyAllTest = {'params': parameters,'data': data}
pickle.dump(numpyAllTest, open('dataNumpyAll.p','wb'))


