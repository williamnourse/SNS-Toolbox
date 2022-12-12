import matplotlib.pyplot as plt
import numpy as np
import pickle

def plot_graph(data, data_brian, data_nengo, index, title, threshold):
    numNeurons = data['shape']

    npRawTimes = data['numpy']
    npAvgTimes = np.mean(npRawTimes, axis=1) * 1000
    npVar = np.std(npRawTimes, axis=1) * 1000

    torchCPURawTimes = data['torchCPU']
    torchCPUAvgTimes = np.mean(torchCPURawTimes, axis=1) * 1000
    torchCPUVar = np.std(torchCPURawTimes, axis=1) * 1000

    torchGPURawTimes = data['torchGPU']
    torchGPUAvgTimes = np.mean(torchGPURawTimes, axis=1) * 1000
    torchGPUVar = np.std(torchGPURawTimes, axis=1) * 1000

    sparseCPURawTimes = data['sparseCPU']
    sparseCPUAvgTimes = np.mean(sparseCPURawTimes, axis=1) * 1000
    sparseCPUVar = np.std(sparseCPURawTimes, axis=1) * 1000

    sparseGPURawTimes = data['sparseGPU']
    sparseGPUAvgTimes = np.mean(sparseGPURawTimes, axis=1) * 1000
    sparseGPUVar = np.std(sparseGPURawTimes, axis=1) * 1000

    manualRawTimes = data['manual']
    manualAvgTimes = np.mean(manualRawTimes, axis=1) * 1000
    manualVar = np.std(manualRawTimes, axis=1) * 1000

    brianRawTimes = data_brian['brian']
    brianAvgTimes = np.mean(brianRawTimes, axis=1) * 1000
    brianVar = np.std(brianRawTimes, axis=1) * 1000

    nengoRawTimes = data_nengo['nengo']
    nengoAvgTimes = np.mean(nengoRawTimes, axis=1) * 1000
    nengoVar = np.std(nengoRawTimes, axis=1) * 1000

    """
    ########################################################################################################################
    PLOTTING
    """

    plt.subplot(2,2,index)

    plt.plot(numNeurons, npAvgTimes, color='C0', label='Numpy')
    plt.fill_between(numNeurons, npAvgTimes - npVar, npAvgTimes + npVar, color='C0', alpha=0.2)
    plt.plot(numNeurons, torchCPUAvgTimes, color='C1', label='Torch CPU')
    plt.fill_between(numNeurons, torchCPUAvgTimes - torchCPUVar, torchCPUAvgTimes + torchCPUVar, color='C1', alpha=0.2)
    plt.plot(numNeurons, torchGPUAvgTimes, color='C2', label='Torch GPU')
    plt.fill_between(numNeurons, torchGPUAvgTimes - torchGPUVar, torchGPUAvgTimes + torchGPUVar, color='C2', alpha=0.2)
    plt.plot(numNeurons, sparseCPUAvgTimes, color='C3', label='Sparse CPU')
    plt.fill_between(numNeurons, sparseCPUAvgTimes - sparseCPUVar, sparseCPUAvgTimes + sparseCPUVar, color='C3',
                     alpha=0.2)
    plt.plot(numNeurons, sparseGPUAvgTimes, color='C4', label='Sparse GPU')
    plt.fill_between(numNeurons, sparseGPUAvgTimes - sparseGPUVar, sparseGPUAvgTimes + sparseGPUVar, color='C4',
                     alpha=0.2)
    plt.plot(numNeurons, manualAvgTimes, color='C5', label='Iterative')
    plt.fill_between(numNeurons, manualAvgTimes - manualVar, manualAvgTimes + manualVar, color='C5', alpha=0.2)
    plt.plot(numNeurons, brianAvgTimes, color='C6', label='Brian2')
    plt.fill_between(numNeurons, brianAvgTimes - brianVar, brianAvgTimes + brianVar, color='C6', alpha=0.2)
    plt.plot(numNeurons, nengoAvgTimes, color='C7', label='Nengo')
    plt.fill_between(numNeurons, nengoAvgTimes - nengoVar, nengoAvgTimes + nengoVar, color='C7', alpha=0.2)
    plt.axhline(y=threshold, color='black', label='Real-Time Boundary', linestyle='--')
    plt.xlim([numNeurons[0], numNeurons[-1]])
    plt.xlabel('Number of Neurons')
    plt.ylabel('Step Time (ms)')
    plt.yscale('log')
    plt.xscale('log')
    # plt.xlim([10,3000])
    plt.title(title)
    if index == 1:
        plt.legend()


plt.figure()

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
