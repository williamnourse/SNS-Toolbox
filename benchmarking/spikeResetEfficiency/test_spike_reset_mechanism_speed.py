"""
Speed comparison between iterative spike-generation vs parallelized
William Nourse
August 24th, 2021
Hello Peter
"""

import numpy as np
import pickle
import time
import sys

"""
########################################################################################################################
ITERATIVE METHOD
Iteratively check whether each neuron has spiked, and update the membrane accordingly
"""

def iterative(numSteps,Ulast,timeFactorMembrane,Gm,Ib,thetaLast,timeFactorThreshold,theta0,m,numNeurons,spikes):
    for i in range(numSteps):
        U = Ulast + timeFactorMembrane * (-Gm * Ulast + Ib)
        theta = thetaLast + timeFactorThreshold * (-thetaLast + theta0 + m * Ulast)
        for j in range(numNeurons):
            if spikes[j] == 1.0:
                spikes[j] = 0.0
            if U[j]>theta[j]:
                U[j] = 0
                spikes[j] = 1.0
        # spikes = np.sign(np.minimum(0, theta - u))
        # u = u * (spikes + 1)

        Ulast = np.copy(U)
        thetaLast = np.copy(theta)

def testIterative(netSize,numSteps,percentSpiking):
    Ulast = np.zeros(netSize)
    Cm = np.zeros(netSize) + 5.0  # membrane capacitance (nF)
    Gm = np.zeros(netSize) + 1.0  # membrane conductance (uS)
    Ib = np.zeros(netSize) + 20.0  # bias current (nA)
    m = np.zeros(netSize)
    spikes = np.zeros(netSize)
    dt = 0.001
    tauTheta = 1
    theta0 = np.zeros(netSize)
    i = 0
    while i < (percentSpiking*netSize):
        theta0[i] = 1.0
        i += 1
    while i < netSize:
        theta0[i] = sys.float_info.max
        i += 1
    timeFactorMembrane = dt / Cm  # multiplicative time factor (save on operations)
    timeFactorThreshold = dt / tauTheta
    start = time.time()
    iterative(numSteps,Ulast,timeFactorMembrane,Gm,Ib,theta0,timeFactorThreshold,theta0,m,netSize,spikes)
    stop = time.time()

    return (stop-start)/numSteps

"""
########################################################################################################################
VECTOR METHOD
Update all the membranes at the same time, including the resets
"""

def vector(numSteps,Ulast,timeFactorMembrane,Gm,Ib,thetaLast,timeFactorThreshold,theta0,m):
    for i in range(numSteps):
        U = Ulast + timeFactorMembrane * (-Gm * Ulast + Ib)
        theta = thetaLast + timeFactorThreshold * (-thetaLast + theta0 + m * Ulast)
        spikes = np.sign(np.minimum(0, theta - U))
        U = U * (spikes + 1)

        Ulast = np.copy(U)
        thetaLast = np.copy(theta)

def testVector(netSize,numSteps,percentSpiking):
    Ulast = np.zeros(netSize)
    Cm = np.zeros(netSize) + 5.0  # membrane capacitance (nF)
    Gm = np.zeros(netSize) + 1.0  # membrane conductance (uS)
    Ib = np.zeros(netSize) + 20.0  # bias current (nA)
    m = np.zeros(netSize)
    spikes = np.zeros(netSize)
    dt = 0.001
    tauTheta = 1
    theta0 = np.zeros(netSize)
    i = 0
    while i < (percentSpiking*netSize):
        theta0[i] = 1.0
        i += 1
    while i < netSize:
        theta0[i] = sys.float_info.max
        i += 1
    timeFactorMembrane = dt / Cm  # multiplicative time factor (save on operations)
    timeFactorThreshold = dt / tauTheta
    start = time.time()
    vector(numSteps,Ulast,timeFactorMembrane,Gm,Ib,theta0,timeFactorThreshold,theta0,m)
    stop = time.time()

    return (stop-start)/numSteps

"""
########################################################################################################################
EXECUTION COMPARISONS
Benchmark the two methods, and see which way is faster (and how much faster)
"""

numSamples = 100    # number of timing samples

# Vary network size
netSize = np.logspace(0,5,num=numSamples)
numSteps = 10000
percentSpiking = 0.5
timesIt = np.zeros(len(netSize))
timesVec = np.zeros(len(netSize))
for i in range(len(netSize)):
    print('Vary Size: %d/%d'%(i+1,numSamples))
    timesIt[i] = testIterative(int(netSize[i]), numSteps, percentSpiking)
    timesVec[i] = testVector(int(netSize[i]), numSteps, percentSpiking)

pickleData = {'numSamples':         numSamples,
              'numSteps':           numSteps,
              'sizes':              np.copy(netSize),
              'sizePercentSplit':   percentSpiking,
              'sizeTimesIt':         np.copy(timesIt),
              'sizeTimesVec':         np.copy(timesVec)
              }

# Vary percent spiking
netSize = 1000
numSteps = 10000
percentSpiking = np.linspace(0.0,1.0,num=numSamples)
timesIt = np.zeros(len(percentSpiking))
timesVec = np.zeros(len(percentSpiking))
for i in range(len(percentSpiking)):
    print('Vary Percent: %d/%d'%(i+1,numSamples))
    timesIt[i] = testIterative(netSize, numSteps, percentSpiking[i])
    timesVec[i] = testVector(netSize, numSteps, percentSpiking[i])

pickleData.update({'percentSize':        netSize,
                   'percents':           np.copy(percentSpiking),
                   'percentTimesIt':      np.copy(timesIt),
                   'percentTimesVec':      np.copy(timesVec)})

pickle.dump(pickleData, open('resetMethodComparisonData.p', 'wb'))