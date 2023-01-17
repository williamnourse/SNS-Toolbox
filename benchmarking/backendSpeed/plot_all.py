import matplotlib.pyplot as plt
import numpy as np
import pickle

def plot_graph(data, data_brian, data_nengo, index, title, threshold, lower=5, upper=95):
    numNeurons = data['shape']

    npRawTimes = data['numpy']
    npAvgTimes = np.mean(npRawTimes[:, 1:], axis=1) * 1000
    npVar = np.std(npRawTimes[:, 1:], axis=1) * 1000
    npLow = np.percentile(npRawTimes[:, 1:], lower, axis=1) * 1000
    npHigh = np.percentile(npRawTimes[:, 1:], upper, axis=1) * 1000

    torchCPURawTimes = data['torchCPU']
    torchCPUAvgTimes = np.mean(torchCPURawTimes[:, 1:], axis=1) * 1000
    torchCPUVar = np.std(torchCPURawTimes[:, 1:], axis=1) * 1000
    torchCPULow = np.percentile(torchCPURawTimes[:, 1:], lower, axis=1) * 1000
    torchCPUHigh = np.percentile(torchCPURawTimes[:, 1:], upper, axis=1) * 1000

    torchGPURawTimes = data['torchGPU']
    torchGPUAvgTimes = np.mean(torchGPURawTimes[:, 1:], axis=1) * 1000
    torchGPUVar = np.std(torchGPURawTimes[:, 1:], axis=1) * 1000
    torchGPULow = np.percentile(torchGPURawTimes[:, 1:], lower, axis=1) * 1000
    torchGPUHigh = np.percentile(torchGPURawTimes[:, 1:], upper, axis=1) * 1000

    sparseCPURawTimes = data['sparseCPU']
    sparseCPUAvgTimes = np.mean(sparseCPURawTimes[:, 1:], axis=1) * 1000
    sparseCPUVar = np.std(sparseCPURawTimes[:, 1:], axis=1) * 1000
    sparseCPULow = np.percentile(sparseCPURawTimes[:, 1:], lower, axis=1) * 1000
    sparseCPUHigh = np.percentile(sparseCPURawTimes[:, 1:], upper, axis=1) * 1000

    sparseGPURawTimes = data['sparseGPU']
    sparseGPUAvgTimes = np.mean(sparseGPURawTimes[:, 1:], axis=1) * 1000
    sparseGPUVar = np.std(sparseGPURawTimes[:, 1:], axis=1) * 1000
    sparseGPULow = np.percentile(sparseCPURawTimes[:, 1:], lower, axis=1) * 1000
    sparseGPUHigh = np.percentile(sparseCPURawTimes[:, 1:], upper, axis=1) * 1000

    manualRawTimes = data['manual']
    manualAvgTimes = np.mean(manualRawTimes[:, 1:], axis=1) * 1000
    manualVar = np.std(manualRawTimes[:, 1:], axis=1) * 1000
    manualLow = np.percentile(manualRawTimes[:, 1:], lower, axis=1) * 1000
    manualHigh = np.percentile(manualRawTimes[:, 1:], upper, axis=1) * 1000

    brianRawTimes = data_brian['brian']
    brianAvgTimes = np.mean(brianRawTimes, axis=1) * 1000
    brianVar = np.std(brianRawTimes, axis=1) * 1000
    brianLow = np.percentile(brianRawTimes[:, 1:], lower, axis=1) * 1000
    brianHigh = np.percentile(brianRawTimes[:, 1:], upper, axis=1) * 1000

    nengoRawTimes = data_nengo['nengo']
    nengoAvgTimes = np.mean(nengoRawTimes, axis=1) * 1000
    nengoVar = np.std(nengoRawTimes, axis=1) * 1000
    nengoLow = np.percentile(nengoRawTimes[:, 1:], lower, axis=1) * 1000
    nengoHigh = np.percentile(nengoRawTimes[:, 1:], upper, axis=1) * 1000

    """
    ########################################################################################################################
    PLOTTING
    """

    plt.figure()
    linestyle = 'solid'
    plt.plot(numNeurons, npAvgTimes, color='C0', label='Numpy', linestyle=linestyle)
    plt.fill_between(numNeurons, npLow, npHigh, color='C0', alpha=0.2, linestyle=linestyle)
    plt.plot(numNeurons, torchCPUAvgTimes, color='C1', label='Torch CPU', linestyle=linestyle)
    plt.fill_between(numNeurons, torchCPULow, torchCPUHigh + torchCPUVar, color='C1', alpha=0.2, linestyle=linestyle)
    plt.plot(numNeurons, torchGPUAvgTimes, color='C2', label='Torch GPU', linestyle=linestyle)
    plt.fill_between(numNeurons, torchGPULow, torchGPUHigh, color='C2', alpha=0.2, linestyle=linestyle)
    plt.plot(numNeurons, sparseCPUAvgTimes, color='C3', label='Sparse CPU', linestyle=linestyle)
    plt.fill_between(numNeurons, sparseCPULow, sparseCPUHigh, color='C3',
                     alpha=0.2, linestyle=linestyle)
    plt.plot(numNeurons, sparseGPUAvgTimes, color='C4', label='Sparse GPU', linestyle=linestyle)
    plt.fill_between(numNeurons, sparseGPULow, sparseGPUHigh, color='C4',
                     alpha=0.2, linestyle=linestyle)
    plt.plot(numNeurons, manualAvgTimes, color='C5', label='Iterative', linestyle=linestyle)
    plt.fill_between(numNeurons, manualLow, manualHigh, color='C5', alpha=0.2, linestyle=linestyle)
    plt.plot(numNeurons, brianAvgTimes, color='C6', label='Brian2')
    plt.fill_between(numNeurons, brianLow, brianHigh, color='C6', alpha=0.2)
    plt.plot(numNeurons, nengoAvgTimes, color='C7', label='Nengo')
    plt.fill_between(numNeurons, nengoLow, nengoHigh, color='C7', alpha=0.2)
    plt.axhline(y=threshold, color='black', label='Real-Time Boundary', linestyle='--')
    plt.xlim([numNeurons[0], numNeurons[-1]])
    plt.xlabel('Number of Neurons')
    plt.ylabel('Step Time (ms)')
    plt.yscale('log')
    plt.xscale('log')
    # plt.xlim([10,3000])
    plt.title(title)
    plt.legend()


data = pickle.load(open('dataBackendTimesNonspikingDense.p', 'rb'))
data_brian = pickle.load(open('dataBrianTimesNonspikingDense.p', 'rb'))
data_nengo = pickle.load(open('dataNengoTimesNonspikingDense.p','rb'))
title = 'Nonspiking Dense Networks'

plot_graph(data, data_brian, data_nengo, 1, title, 1)

data = pickle.load(open('dataBackendTimesNonspikingSparse.p', 'rb'))
data_brian = pickle.load(open('dataBrianTimesNonspikingSparse.p', 'rb'))
data_nengo = pickle.load(open('dataNengoTimesNonspikingSparse.p','rb'))
title = 'Nonspiking Sparse Networks'

plot_graph(data, data_brian, data_nengo, 2, title, 1)

data = pickle.load(open('dataBackendTimesSpikingDense.p', 'rb'))
data_brian = pickle.load(open('dataBrianTimesSpikingDense.p', 'rb'))
data_nengo = pickle.load(open('dataNengoTimesSpikingDense.p','rb'))
title = 'Spiking Dense Networks'

plot_graph(data, data_brian, data_nengo, 3, title, 0.1)

data = pickle.load(open('dataBackendTimes.p', 'rb'))
data_brian = pickle.load(open('dataBrianTimesSpikingSparse.p', 'rb'))
data_nengo = pickle.load(open('dataNengoTimesSpikingSparse.p','rb'))
title = 'Spiking Sparse Networks'

plot_graph(data, data_brian, data_nengo, 4, title, 0.1)

plt.show()
