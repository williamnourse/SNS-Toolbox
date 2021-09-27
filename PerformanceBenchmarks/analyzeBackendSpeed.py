"""
Analyze the simulation speed of the different backends
William Nourse
September 9 2021
Little hand says it's time to rock and roll
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle

# this can be found inside 'performanceData.zip'
data = pickle.load(open('dataBackendTimes.p', 'rb'))
numNeurons = data['numNeurons']

npRawTimes = data['numpy']
npAvgTimes = np.mean(npRawTimes,axis=1)*1000
npVar = np.std(npRawTimes,axis=1)*1000

torchCPURawTimes = data['torchCPU']
torchCPUAvgTimes = np.mean(torchCPURawTimes,axis=1)*1000
torchCPUVar = np.std(torchCPURawTimes,axis=1)*1000

torchGPURawTimes = data['torchGPU']
torchGPUAvgTimes = np.mean(torchGPURawTimes,axis=1)*1000
torchGPUVar = np.std(torchGPURawTimes,axis=1)*1000

torchGPUTransferRawTimes = data['torchGPUTransfer']
torchGPUTransferAvgTimes = np.mean(torchGPUTransferRawTimes,axis=1)*1000
torchGPUTransferVar = np.std(torchGPURawTimes,axis=1)*1000

torchGPUSparseRawTimes = data['torchGPUSparse']
torchGPUSparseAvgTimes = np.mean(torchGPUSparseRawTimes,axis=1)*1000
torchGPUSparseVar = np.std(torchGPUSparseRawTimes,axis=1)*1000


"""
########################################################################################################################
PLOTTING
"""

plt.figure()

plt.plot(numNeurons,npAvgTimes,color='C0',label='Numpy')
plt.fill_between(numNeurons,npAvgTimes-npVar,npAvgTimes+npVar,color='C0',alpha=0.2)
plt.plot(numNeurons,torchCPUAvgTimes,color='C1',label='Torch CPU')
plt.fill_between(numNeurons,torchCPUAvgTimes-torchCPUVar,torchCPUAvgTimes+torchCPUVar,color='C1',alpha=0.2)
plt.plot(numNeurons,torchGPUAvgTimes,color='C2',label='Torch GPU')
plt.fill_between(numNeurons,torchGPUAvgTimes-torchGPUVar,torchGPUAvgTimes+torchGPUVar,color='C2',alpha=0.2)
plt.plot(numNeurons,torchGPUTransferAvgTimes,color='C3',label='Torch GPU Transfer')
plt.fill_between(numNeurons,torchGPUTransferAvgTimes-torchGPUTransferVar,torchGPUTransferAvgTimes+torchGPUTransferVar,color='C3',alpha=0.2)
plt.plot(numNeurons,torchGPUSparseAvgTimes,color='C4',label='Torch GPU Sparse')
plt.fill_between(numNeurons,torchGPUSparseAvgTimes-torchGPUSparseVar,torchGPUSparseAvgTimes+torchGPUSparseVar,color='C3',alpha=0.2)
plt.xlabel('Number of Neurons')
plt.ylabel('Step Time (ms)')
plt.yscale('log')
plt.xscale('log')
plt.xlim([10,3000])
plt.title('Average Simulation Step Time as a Function of Network Size')
plt.legend()

plt.show()