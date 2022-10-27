"""
Analyze the simulation speed of the different backends
William Nourse
September 9 2021
Little hand says it's time to rock and roll
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle

# this can be found inside 'backendSpeedData.zip'
# data = pickle.load(open('backendSpeedData/dataBackendTimesNew.p', 'rb'))
data = pickle.load(open('dataBackendTimes.p', 'rb'))
numNeurons = data['shape']

npRawTimes = data['numpy']
npAvgTimes = np.mean(npRawTimes,axis=1)*1000
npVar = np.std(npRawTimes,axis=1)*1000

torchCPURawTimes = data['torchCPU']
torchCPUAvgTimes = np.mean(torchCPURawTimes,axis=1)*1000
torchCPUVar = np.std(torchCPURawTimes,axis=1)*1000

torchGPURawTimes = data['torchGPU']
torchGPUAvgTimes = np.mean(torchGPURawTimes,axis=1)*1000
torchGPUVar = np.std(torchGPURawTimes,axis=1)*1000

sparseCPURawTimes = data['sparseCPU']
sparseCPUAvgTimes = np.mean(sparseCPURawTimes,axis=1)*1000
sparseCPUVar = np.std(sparseCPURawTimes,axis=1)*1000

sparseGPURawTimes = data['sparseGPU']
sparseGPUAvgTimes = np.mean(sparseGPURawTimes,axis=1)*1000
sparseGPUVar = np.std(sparseGPURawTimes,axis=1)*1000

manualRawTimes = data['manual']
manualAvgTimes = np.mean(manualRawTimes,axis=1)*1000
manualVar = np.std(manualRawTimes,axis=1)*1000


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
plt.plot(numNeurons,sparseCPUAvgTimes,color='C3',label='Sparse CPU')
plt.fill_between(numNeurons,sparseCPUAvgTimes-sparseCPUVar,sparseCPUAvgTimes+sparseCPUVar,color='C3',alpha=0.2)
plt.plot(numNeurons,sparseGPUAvgTimes,color='C4',label='Sparse GPU')
plt.fill_between(numNeurons,sparseGPUAvgTimes-sparseGPUVar,sparseGPUAvgTimes+sparseGPUVar,color='C4',alpha=0.2)
plt.plot(numNeurons,manualAvgTimes,color='C5',label='Manual')
plt.fill_between(numNeurons,manualAvgTimes-manualVar,manualAvgTimes+manualVar,color='C5',alpha=0.2)
plt.xlabel('Number of Neurons')
plt.ylabel('Step Time (ms)')
plt.yscale('log')
plt.xscale('log')
# plt.xlim([10,3000])
plt.title('Average Simulation Step Time as a Function of Network Size')
plt.legend()

plt.show()