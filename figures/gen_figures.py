import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sea

def plot_shaded(neurons, raw_times, color, label, lower=5, upper=95, linestyle=None, alpha=0.2):
    avg = np.mean(raw_times[:, 1:], axis=1) * 1000
    low = np.percentile(raw_times[:, 1:], lower, axis=1) * 1000
    high = np.percentile(raw_times[:, 1:], upper, axis=1) * 1000
    if linestyle is None:
        linestyle = 'solid'
    plt.plot(neurons, avg, color=color, linestyle=linestyle, label=label)
    plt.fill_between(neurons, low, high, color=color, alpha=alpha, linestyle=linestyle)

def plot_multiple_shaded(neurons, title, raw_times_list, colors, labels, linestyles=None, lower=5, upper=95, alpha=0.2, legend=True):
    if linestyles is None:
        for i in range(len(raw_times_list)):
            plot_shaded(neurons, raw_times_list[i], colors[i], labels[i], lower=lower, upper=upper, alpha=alpha)
    else:
        for i in range(len(raw_times_list)):
            plot_shaded(neurons, raw_times_list[i], colors[i], labels[i], lower=lower, upper=upper, linestyle=linestyles[i], alpha=alpha)
    plt.xlim([neurons[0], neurons[-1]])
    plt.xlabel('Number of Neurons')
    plt.ylabel('Step Time (ms)')
    plt.yscale('log')
    plt.xscale('log')
    plt.title(title)
    if legend:
        plt.legend()

dataNonspikingDense = pickle.load(open('../benchmarking/backendSpeed/dataBackendTimesNonspikingDense.p', 'rb'))
dataBrianNonspikingDense = pickle.load(open('../benchmarking/backendSpeed/dataBrianTimesNonspikingDense.p', 'rb'))
dataBrianCUDANonspikingDense = pickle.load(open('../benchmarking/backendSpeed/dataBrianCudaTimesNonspikingDense.p', 'rb'))
dataNengoNonspikingDense = pickle.load(open('../benchmarking/backendSpeed/dataNengoTimesNonspikingDense.p', 'rb'))
dataANNarchyNonspikingDense = pickle.load(open('../benchmarking/otherSoftware/annarchy/dataANNarchyTimesNonspikingDense.p', 'rb'))
dataANNarchyCUDANonspikingDense = pickle.load(open('../benchmarking/otherSoftware/annarchy/dataANNarchyCUDATimesNonspikingDense.p', 'rb'))

neurons = dataNonspikingDense['shape']
raw_times = [dataNonspikingDense['numpy'], dataNonspikingDense['torchCPU'], dataNonspikingDense['torchGPU'],
             dataNonspikingDense['sparseCPU'], dataNonspikingDense['sparseGPU'], dataNonspikingDense['manual'],
             dataBrianNonspikingDense['brian'], dataBrianCUDANonspikingDense['brian'], dataNengoNonspikingDense['nengo'],
             dataANNarchyNonspikingDense['annarchy'], dataANNarchyCUDANonspikingDense['annarchy']]
colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10']
labels = ['Numpy', 'Torch (CPU)', 'Torch (GPU)', 'Sparse (CPU)', 'Sparse (GPU)', 'Iterative', 'Brian2', 'Brian2CUDA', 'Nengo', 'ANNarchy (CPU)', 'ANNarchy (GPU)']
linestyles = ['solid', '--', 'solid', '--','solid', '--','solid', '--','solid', '--',':']

plt.figure()
plot_multiple_shaded(neurons, 'Nonspiking Dense', raw_times, colors, labels, linestyles=linestyles)

plt.show()
