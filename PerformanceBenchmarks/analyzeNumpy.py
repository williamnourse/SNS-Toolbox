"""
Read the data collected in the benchmarking test
William Nourse
August 26th, 2021
Wen Moon?
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

"""
########################################################################################################################
ALL COMPONENTS
"""
output = pickle.load(open('dataNumpyAll.p','rb'))
params = output['params']
networkSize = params['networkSize']
percentSpiking = params['percentSpiking']
probConnectivity = params['probConnectivity']
dataContainer = output['data']
rawData = dataContainer['data']

avgStepAll = np.zeros([len(rawData), len(rawData[0]), len(rawData[0][0])])
stdStepAll = np.zeros([len(rawData), len(rawData[0]), len(rawData[0][0])])
for size in range(len(rawData)):
    for perSpike in range(len(rawData[0])):
        for probConn in range(len(rawData[0][0])):
            avgStepAll[size][perSpike][probConn] = np.mean(rawData[size][perSpike][probConn]*1000)
            stdStepAll[size][perSpike][probConn] = np.std(rawData[size][perSpike][probConn]*1000)

"""
########################################################################################################################
NO REFRACTORY PERIOD
"""
output = pickle.load(open('dataNumpyNoRef.p','rb'))
params = output['params']
networkSize = params['networkSize']
percentSpiking = params['percentSpiking']
probConnectivity = params['probConnectivity']
dataContainer = output['data']
rawData = dataContainer['data']

avgStepNoRef = np.zeros([len(rawData), len(rawData[0]), len(rawData[0][0])])
stdStepNoRef = np.zeros([len(rawData), len(rawData[0]), len(rawData[0][0])])
for size in range(len(rawData)):
    for perSpike in range(len(rawData[0])):
        for probConn in range(len(rawData[0][0])):
            avgStepNoRef[size][perSpike][probConn] = np.mean(rawData[size][perSpike][probConn]*1000)
            stdStepNoRef[size][perSpike][probConn] = np.std(rawData[size][perSpike][probConn]*1000)

"""
########################################################################################################################
NO SPIKING
"""
output = pickle.load(open('dataNumpyNoSpike.p','rb'))
params = output['params']
networkSize = params['networkSize']
probConnectivity = params['probConnectivity']
dataContainer = output['data']
rawData = dataContainer['data']

avgStepNoSpike = np.zeros([len(rawData), len(rawData[0])])
stdStepNoSpike = np.zeros([len(rawData), len(rawData[0])])
for size in range(len(rawData)):
    for probConn in range(len(rawData[0])):
        avgStepNoSpike[size][probConn] = np.mean(rawData[size][probConn]*1000)
        stdStepNoSpike[size][probConn] = np.std(rawData[size][probConn]*1000)


plt.figure()
plt.plot(networkSize, avgStepAll[:][-1][-1],color='tab:blue',label='Full Model')
plt.fill_between(networkSize,avgStepAll[:][-1][-1] - stdStepAll[:][-1][-1],avgStepAll[:][-1][-1] + stdStepAll[:][-1][-1],alpha=0.2,color='tab:blue')
plt.plot(networkSize, avgStepNoRef[:][-1][-1],color='tab:orange',label='No Refractory Period')
plt.fill_between(networkSize,avgStepNoRef[:][-1][-1] - stdStepNoRef[:][-1][-1],avgStepNoRef[:][-1][-1] + stdStepNoRef[:][-1][-1],alpha=0.2,color='tab:orange')
plt.plot(networkSize, avgStepNoSpike[:][-1],color='tab:green',label='No Spiking Mechanisms')
plt.fill_between(networkSize,avgStepNoSpike[:][-1] - stdStepNoSpike[:][-1],avgStepNoSpike[:][-1] + stdStepNoSpike[:][-1],alpha=0.2,color='tab:green')
plt.xlabel('Number of Neurons')
plt.ylabel('Time for 1 Step (ms)')
plt.legend()
plt.show()