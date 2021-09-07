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
rawDataAll = dataContainer['data']

avgStepAll = np.zeros([len(rawDataAll), len(rawDataAll[0][0])])
stdStepAll = np.zeros([len(rawDataAll), len(rawDataAll[0][0])])
for size in range(len(rawDataAll)):
    for probConn in range(len(rawDataAll[0][0])):
        avgStepAll[size][probConn] = np.mean(rawDataAll[size][0][probConn]*1000)
        stdStepAll[size][probConn] = np.std(rawDataAll[size][0][probConn]*1000)

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
rawDataNoRef = dataContainer['data']

avgStepNoRef = np.zeros([len(rawDataNoRef), len(rawDataNoRef[0][0])])
stdStepNoRef = np.zeros([len(rawDataNoRef), len(rawDataNoRef[0][0])])
for size in range(len(rawDataNoRef)):
    for probConn in range(len(rawDataNoRef[0][0])):
        avgStepNoRef[size][probConn] = np.mean(rawDataNoRef[size][0][probConn]*1000)
        stdStepNoRef[size][probConn] = np.std(rawDataNoRef[size][0][probConn]*1000)

"""
########################################################################################################################
NO SPIKING
"""
output = pickle.load(open('dataNumpyNoSpike.p','rb'))
params = output['params']
networkSize = params['networkSize']
probConnectivity = params['probConnectivity']
dataContainer = output['data']
rawDataNoSpike = dataContainer['data']

avgStepNoSpike = np.zeros([len(rawDataNoSpike), len(rawDataNoSpike[0])])
stdStepNoSpike = np.zeros([len(rawDataNoSpike), len(rawDataNoSpike[0])])
for size in range(len(rawDataNoSpike)):
    for probConn in range(len(rawDataNoSpike[0])):
        avgStepNoSpike[size][probConn] = np.mean(rawDataNoSpike[size][probConn]*1000)
        stdStepNoSpike[size][probConn] = np.std(rawDataNoSpike[size][probConn]*1000)

networkSize = networkSize.astype(int)

plt.figure()
i = 0
while i < len(probConnectivity):
    plt.subplot(2,5,i+1)
    plt.plot(networkSize, avgStepAll[:,i],color='tab:blue',label='Full Model')
    plt.fill_between(networkSize,avgStepAll[:,i] - stdStepAll[:,i],avgStepAll[:,i] + stdStepAll[:,i],alpha=0.2,color='tab:blue')
    plt.plot(networkSize, avgStepNoRef[:,i],color='tab:orange',label='No Refractory Period')
    plt.fill_between(networkSize,avgStepNoRef[:,i] - stdStepNoRef[:,i],avgStepNoRef[:,i] + stdStepNoRef[:,i],alpha=0.2,color='tab:orange')
    plt.plot(networkSize, avgStepNoSpike[:,i],color='tab:green',label='No Spiking Mechanisms')
    plt.fill_between(networkSize,avgStepNoSpike[:,i] - stdStepNoSpike[:,i],avgStepNoSpike[:,i] + stdStepNoSpike[:,i],alpha=0.2,color='tab:green')
    plt.xlabel('Number of Neurons')
    plt.ylabel('Avg Time for 1 Step (ms)')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('%d %% Connectivity' % (probConnectivity[i]*100))
    plt.legend()
    i+=1

plt.figure()
plt.subplot(3,1,1)
for i in range(len(probConnectivity)):
    plt.plot(networkSize, avgStepAll[:, i],color=str('C%i'%i),label=str('%d %% Connectivity' % (probConnectivity[i]*100)))
    plt.fill_between(networkSize, avgStepAll[:, i] - stdStepAll[:, i], avgStepAll[:, i] + stdStepAll[:, i], alpha=0.2,
                     color=str('C%i'%i))
plt.title('All Components')
plt.xlabel('Number of Neurons')
plt.ylabel('Avg Time for 1 Step (ms)')
plt.xscale('log')
plt.yscale('log')
plt.legend()

plt.subplot(3,1,2)
for i in range(len(probConnectivity)):
    plt.plot(networkSize, avgStepNoRef[:, i],color=str('C%i'%i),label=str('%d %% Connectivity' % (probConnectivity[i]*100)))
    plt.fill_between(networkSize, avgStepNoRef[:, i] - stdStepNoRef[:, i], avgStepNoRef[:, i] + stdStepNoRef[:, i], alpha=0.2,
                     color=str('C%i'%i))
plt.title('No Refractory Period')
plt.xlabel('Number of Neurons')
plt.ylabel('Avg Time for 1 Step (ms)')
plt.xscale('log')
plt.yscale('log')
plt.legend()

plt.subplot(3,1,3)
for i in range(len(probConnectivity)):
    plt.plot(networkSize, avgStepNoSpike[:, i],color=str('C%i'%i),label=str('%d %% Connectivity' % (probConnectivity[i]*100)))
    plt.fill_between(networkSize, avgStepNoSpike[:, i] - stdStepNoSpike[:, i], avgStepNoSpike[:, i] + stdStepNoSpike[:, i], alpha=0.2,
                     color=str('C%i'%i))
plt.title('No Spiking Mechanism')
plt.xlabel('Number of Neurons')
plt.ylabel('Avg Time for 1 Step (ms)')
plt.xscale('log')
plt.yscale('log')
plt.legend()

plt.show()