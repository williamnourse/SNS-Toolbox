import matplotlib.pyplot as plt
import numpy as np
import pickle

def plot_graph(data, threshold, title, linestyle, lower=5, upper=95):
    numNeurons = data['shape']

    npRawTimes = data['numpy']
    npAvgTimes = np.mean(npRawTimes[:,1:], axis=1) * 1000
    npVar = np.std(npRawTimes[:,1:], axis=1) * 1000
    npLow = np.percentile(npRawTimes[:,1:], lower, axis=1) * 1000
    npHigh = np.percentile(npRawTimes[:,1:], upper, axis=1) * 1000

    torchCPURawTimes = data['torchCPU']
    torchCPUAvgTimes = np.mean(torchCPURawTimes[:,1:], axis=1) * 1000
    torchCPUVar = np.std(torchCPURawTimes[:,1:], axis=1) * 1000
    torchCPULow = np.percentile(torchCPURawTimes[:,1:], lower, axis=1) * 1000
    torchCPUHigh = np.percentile(torchCPURawTimes[:,1:], upper, axis=1) * 1000

    torchGPURawTimes = data['torchGPU']
    torchGPUAvgTimes = np.mean(torchGPURawTimes[:,1:], axis=1) * 1000
    torchGPUVar = np.std(torchGPURawTimes[:,1:], axis=1) * 1000
    torchGPULow = np.percentile(torchGPURawTimes[:,1:], lower, axis=1) * 1000
    torchGPUHigh = np.percentile(torchGPURawTimes[:,1:], upper, axis=1) * 1000

    sparseCPURawTimes = data['sparseCPU']
    sparseCPUAvgTimes = np.mean(sparseCPURawTimes[:,1:], axis=1) * 1000
    sparseCPUVar = np.std(sparseCPURawTimes[:,1:], axis=1) * 1000
    sparseCPULow = np.percentile(sparseCPURawTimes[:,1:], lower, axis=1) * 1000
    sparseCPUHigh = np.percentile(sparseCPURawTimes[:,1:], upper, axis=1) * 1000

    sparseGPURawTimes = data['sparseGPU']
    sparseGPUAvgTimes = np.mean(sparseGPURawTimes[:,1:], axis=1) * 1000
    sparseGPUVar = np.std(sparseGPURawTimes[:,1:], axis=1) * 1000
    sparseGPULow = np.percentile(sparseCPURawTimes[:,1:], lower, axis=1) * 1000
    sparseGPUHigh = np.percentile(sparseCPURawTimes[:,1:], upper, axis=1) * 1000

    manualRawTimes = data['manual']
    manualAvgTimes = np.mean(manualRawTimes[:,1:], axis=1) * 1000
    manualVar = np.std(manualRawTimes[:,1:], axis=1) * 1000
    manualLow = np.percentile(manualRawTimes[:,1:], lower, axis=1) * 1000
    manualHigh = np.percentile(manualRawTimes[:,1:], upper, axis=1) * 1000

    """
    ########################################################################################################################
    PLOTTING
    """
    # plt.plot(numNeurons, npAvgTimes, color='C0', label='Numpy', linestyle=linestyle)
    # plt.fill_between(numNeurons, npLow, npHigh, color='C0', alpha=0.2, linestyle=linestyle)
    plt.plot(numNeurons, torchCPUAvgTimes, color='C1', label='Torch CPU', linestyle=linestyle)
    plt.fill_between(numNeurons, torchCPULow, torchCPUHigh + torchCPUVar, color='C1', alpha=0.2, linestyle=linestyle)
    plt.plot(numNeurons, torchGPUAvgTimes, color='C2', label='Torch GPU', linestyle=linestyle)
    plt.fill_between(numNeurons, torchGPULow, torchGPUHigh, color='C2', alpha=0.2, linestyle=linestyle)
    # plt.plot(numNeurons, sparseCPUAvgTimes, color='C3', label='Sparse CPU', linestyle=linestyle)
    # plt.fill_between(numNeurons, sparseCPULow, sparseCPUHigh, color='C3',
    #                  alpha=0.2, linestyle=linestyle)
    # plt.plot(numNeurons, sparseGPUAvgTimes, color='C4', label='Sparse GPU', linestyle=linestyle)
    # plt.fill_between(numNeurons, sparseGPULow, sparseGPUHigh, color='C4',
    #                  alpha=0.2, linestyle=linestyle)
    # plt.plot(numNeurons, manualAvgTimes, color='C5', label='Iterative', linestyle=linestyle)
    # plt.fill_between(numNeurons, manualLow, manualHigh, color='C5', alpha=0.2, linestyle=linestyle)
    plt.axhline(y=threshold, color='black', label='Real-Time Boundary', linestyle='--')
    plt.xlim([numNeurons[0], numNeurons[-1]])
    plt.xlabel('Number of Neurons')
    plt.ylabel('Step Time (ms)')
    plt.yscale('log')
    plt.xscale('log')
    # plt.xlim([10,3000])
    plt.title(title)
    plt.legend()

data_sparse_nonspiking_jetson = pickle.load(open('jetson_nano/dataJetsonNanoNonspikingSparse.p', 'rb'))
data_sparse_spiking_jetson = pickle.load(open('jetson_nano/dataJetsonNanoSpikingSparse.p', 'rb'))
data_dense_nonspiking_jetson = pickle.load(open('jetson_nano/dataJetsonNonspikingDense.p', 'rb'))
data_dense_spiking_jetson = pickle.load(open('jetson_nano/dataJetsonNanoSpikingDense.p', 'rb'))

data_sparse_nonspiking_rpi = pickle.load(open('rpi_3b/dataRpi3bNonspikingSparse.p', 'rb'))
data_sparse_spiking_rpi = pickle.load(open('rpi_3b/dataRpi3bSpikingSparse.p', 'rb'))
data_dense_nonspiking_rpi = pickle.load(open('rpi_3b/dataRpi3bNonspikingDense.p', 'rb'))
data_dense_spiking_rpi = pickle.load(open('rpi_3b/dataRpi3bSpikingDense.p', 'rb'))

data_sparse_nonspiking_nuc = pickle.load(open('intel_nuc/dataNUCNonspikingSparse.p', 'rb'))
data_sparse_spiking_nuc = pickle.load(open('intel_nuc/dataNUCSpikingSparse.p', 'rb'))
data_dense_nonspiking_nuc = pickle.load(open('intel_nuc/dataNUCNonspikingDense.p', 'rb'))
data_dense_spiking_nuc = pickle.load(open('intel_nuc/dataNUCSpikingDense.p', 'rb'))

data_sparse_nonspiking_orin = pickle.load(open('jetson_orin_nano/dataJetsonNanoNonspikingSparse.p', 'rb'))
data_sparse_spiking_orin = pickle.load(open('jetson_orin_nano/dataJetsonNanoSpikingSparse.p', 'rb'))
data_dense_nonspiking_orin = pickle.load(open('jetson_orin_nano/dataJetsonNonspikingDense.p', 'rb'))
data_dense_spiking_orin = pickle.load(open('jetson_orin_nano/dataJetsonNanoSpikingDense.p', 'rb'))

plt.figure()
# plot_graph(data_dense_spiking_jetson,0.1,'Dense Spiking', 'solid')
# plot_graph(data_dense_spiking_rpi,0.1,'Dense Spiking', '--')
# plot_graph(data_dense_spiking_nuc,0.1,'Dense Spiking', ':')
plot_graph(data_dense_spiking_orin,0.1,'Dense Spiking', '-.')

plt.figure()
# plot_graph(data_sparse_spiking_jetson,0.1,'Sparse Spiking', 'solid')
# plot_graph(data_sparse_spiking_rpi,0.1,'Sparse Spiking', '--')
# plot_graph(data_sparse_spiking_nuc,0.1,'Sparse Spiking', ':')
plot_graph(data_sparse_spiking_orin,0.1,'Sparse Spiking', '-.')

plt.figure()
# plot_graph(data_dense_nonspiking_jetson,5,'Dense Nonspiking', 'solid')
# plot_graph(data_dense_nonspiking_rpi,5,'Dense Nonspiking', '--')
# plot_graph(data_dense_nonspiking_nuc,5,'Dense Nonspiking', ':')
plot_graph(data_dense_nonspiking_orin,5,'Dense Nonspiking', '-.')

plt.figure()
# plot_graph(data_sparse_nonspiking_jetson,5,'Sparse Nonspiking', 'solid')
# plot_graph(data_sparse_nonspiking_rpi,5,'Sparse Nonspiking', '--')
# plot_graph(data_sparse_nonspiking_nuc,5,'Sparse Nonspiking', ':')
plot_graph(data_sparse_nonspiking_orin,5,'Sparse Nonspiking', '-.')

plt.show()
