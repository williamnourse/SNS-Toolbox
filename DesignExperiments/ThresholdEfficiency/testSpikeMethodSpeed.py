"""
Tester to figure which of two spiking methods is faster
William Nourse
August 23rd, 2021
The word today is 'Squatter's Rights'
"""
import pickle

import numpy as np
import time

"""
########################################################################################################################
SPIKING METHOD 1
Check if each neuron is a 'spiker' or not (nonspiking and spiking neurons are implemented differently)
"""

def spike1(spikeMask,net,numSteps,theta,thetaLast,spikes):
    """
    Spiking and NonSpiking are handled differently
    :param spikeMask:   binary vector, 1==spiking
    :param net:         vector of neural states
    :param numSteps:    number of simulation steps
    :param theta:       vector of neural thresholds
    :param thetaLast:   vector of neural thresholds at the previous timestep
    :param spikes:      vector of spikes at the current timestep
    :return:
    """
    for step in range(numSteps):    # run for the specified number of steps
        for i in range(len(net)):   # access each neuron
            if spikeMask[i] > 0:    # if spikeMask[i] is 1, then the neuron spikes
                theta[i] = thetaLast[i] + 1.0*(-thetaLast[i]+1.0+1.0*net[i])    # filler for threshold dynamics
                if spikeMask[i] > 0:    # filler for checking if the membrane is above threshold
                    spikes[i] = 1   # write a spike
                    net[i] = 0  # reset the membrane

def testSpike1(netSize,numSteps,percentSpiking):
    """
    Run spiking method 1 following certain constraints
    :param netSize:     number of neurons in the network
    :param numSteps:    number of simulation steps
    :param percentSpiking:  percent of network which is spiking (between 0 and 1)
    :return: the average execution time for a single timestep
    """
    spikeMask = np.zeros(netSize)   # create a vector which stores if a neuron is spiking
    i = 0   # initialize counter
    while i < (netSize*percentSpiking): # set the first chunk of the mask as spiking
        spikeMask[i] = 1    # spiking means the value must be 1
        i += 1  # increment the counter

    # create vectors to represent the different variables
    net = np.zeros(netSize)
    theta = np.zeros(netSize)
    thetaLast = np.zeros(netSize)
    spikes = np.zeros(netSize)

    start = time.time() # start a timer
    spike1(spikeMask,net,numSteps,theta,thetaLast,spikes)   # run the method
    end = time.time()   # stop the timer
    tDiff = end-start   # get the elapsed time

    return tDiff/numSteps


"""
########################################################################################################################
SPIKING METHOD 2
Assume all neurons are spiking (NonSpiking neurons just have an impossibly high threshold
"""


def spike2(spikeMask, net, numSteps, theta, thetaLast, spikes):
    """
    Spiking and NonSpiking are handled the same
    :param spikeMask:   binary vector, 1==spiking
    :param net:         vector of neural states
    :param numSteps:    number of simulation steps
    :param theta:       vector of neural thresholds
    :param thetaLast:   vector of neural thresholds at the previous timestep
    :param spikes:      vector of spikes at the current timestep
    :return:
    """
    for step in range(numSteps):  # run for the specified number of steps
        theta = thetaLast + 1.0 * (-thetaLast + 1.0 + 1.0 * net)  # filler for threshold dynamics
        for i in range(len(net)):
            if spikeMask[i] > 0:  # filler for checking if the membrane is above threshold
                spikes[i] = 1  # write a spike
                net[i] = 0  # reset the membrane


def testSpike2(netSize, numSteps, percentSpiking):
    """
    Run spiking method 2 following certain constraints
    :param netSize:     number of neurons in the network
    :param numSteps:    number of simulation steps
    :param percentSpiking:  percent of network which is spiking (between 0 and 1)
    :return: the average execution time for a single timestep
    """
    spikeMask = np.zeros(netSize)  # create a vector which stores if a neuron is spiking
    i = 0  # initialize counter
    while i < (netSize * percentSpiking):  # set the first chunk of the mask as spiking
        spikeMask[i] = 1  # spiking means the value must be 1
        i += 1  # increment the counter

    # create vectors to represent the different variables
    net = np.zeros(netSize)
    theta = np.zeros(netSize)
    thetaLast = np.zeros(netSize)
    spikes = np.zeros(netSize)

    start = time.time()  # start a timer
    spike2(spikeMask, net, numSteps, theta, thetaLast, spikes)  # run the method
    end = time.time()  # stop the timer
    tDiff = end-start  # get the elapsed time

    return tDiff / numSteps

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
times1 = np.zeros(len(netSize))
times2 = np.zeros(len(netSize))
for i in range(len(netSize)):
    print('Vary Size: %d/%d'%(i+1,numSamples))
    times1[i] = testSpike1(int(netSize[i]),numSteps,percentSpiking)
    times2[i] = testSpike2(int(netSize[i]), numSteps, percentSpiking)

min1SizeTime = np.min(times1)
max1SizeTime = np.max(times1)
min2SizeTime = np.min(times2)
max2SizeTime = np.max(times2)

pickleData = {'numSamples':         numSamples,
              'numSteps':           numSteps,
              'sizes':              np.copy(netSize),
              'sizePercentSplit':   percentSpiking,
              'sizeTimes1':         np.copy(times1),
              'sizeTimes2':         np.copy(times2)
              }

# Vary percent spiking
netSize = 1000
numSteps = 10000
percentSpiking = np.linspace(0.0,1.0,num=numSamples)
times1 = np.zeros(len(percentSpiking))
times2 = np.zeros(len(percentSpiking))
for i in range(len(percentSpiking)):
    print('Vary Percent: %d/%d'%(i+1,numSamples))
    times1[i] = testSpike1(netSize,numSteps,percentSpiking[i])
    times2[i] = testSpike2(netSize, numSteps, percentSpiking[i])

min1PercentTime = np.min(times1)
max1PercentTime = np.max(times1)
min2PercentTime = np.min(times2)
max2PercentTime = np.max(times2)

pickleData.update({'percentSize':        netSize,
                   'percents':           np.copy(percentSpiking),
                   'percentTimes1':      np.copy(times1),
                   'percentTimes2':      np.copy(times2)})

pickle.dump(pickleData, open('spikingMethodComparisonData.p', 'wb'))

print('\nSize Stats:')
print('Method 1 Min: '+str(min1SizeTime)+'     Max: '+str(max1SizeTime))
print('Method 2 Min: '+str(min2SizeTime)+'     Max: '+str(max2SizeTime))
print('Minimum Difference: '+str(abs(min1SizeTime-min2SizeTime)))
print('Maximum Difference: '+str(abs(max1SizeTime-max2SizeTime)))

print('\nPercent Stats:')
print('Method 1 Min: '+str(min1PercentTime)+'     Max: '+str(max1PercentTime))
print('Method 2 Min: '+str(min2PercentTime)+'     Max: '+str(max2PercentTime))
print('Minimum Difference: '+str(abs(min1PercentTime-min2PercentTime)))
print('Maximum Difference: '+str(abs(max1PercentTime-max2PercentTime)))
