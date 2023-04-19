import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sea
from PIL import Image
import pandas as pd

def stim2activation(stim):
    """
    converts from a neural potential to a muscle activation between 0 and 1 with a clipped sigmoid curve
    :param stim: MN potential in mV
    :return: act: muscle activation between 0 and 1
    """

    steepness = 0.1532
    x_off = -70
    y_offset = -0.01
    amp = 1
    act = amp/(1 + np.exp(steepness*(x_off-stim))) + y_offset
    act = np.clip(act, 0,1)
    return act

def set_zero_to_nan_2d(data):
    shape = np.shape(data)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if data[i,j] == 0.0:
                data[i,j] = np.nan
    return data

def get_best(neurons, raw_times_list):
    new_times = np.zeros_like(raw_times_list[0])
    for i in range(len(raw_times_list)):
        if i == 0:
            avg = np.mean(raw_times_list[i][:, 1:], axis=1)
        else:
            avg = np.vstack((avg,np.mean(raw_times_list[i][:, 1:], axis=1)))
    best_ind = np.argmin(avg,axis=0)
    for i in range(len(neurons)):
        new_times[i,:] = raw_times_list[best_ind[i]][i,:]
    return new_times

def plot_shaded(neurons, raw_times, color, label, lower=5, upper=95, linestyle=None, alpha=0.2):
    avg = np.mean(raw_times[:, 1:], axis=1) * 1000
    low = np.percentile(raw_times[:, 1:], lower, axis=1) * 1000
    high = np.percentile(raw_times[:, 1:], upper, axis=1) * 1000
    if linestyle is None:
        linestyle = 'solid'
    plt.plot(neurons, avg, color=color, linestyle=linestyle, label=label)
    plt.fill_between(neurons, low, high, color=color, alpha=alpha, linestyle=linestyle)

def plot_multiple_shaded(neurons, title, raw_times_list, colors, labels, linestyles, lower=5, upper=95, alpha=0.2, legend=False, xlabel=True, rt=1.0):
    for i in range(len(raw_times_list)):
        plot_shaded(neurons, raw_times_list[i], colors[i], labels[i], lower=lower, upper=upper, linestyle=linestyles[i], alpha=alpha)
    plt.xlim([neurons[0], neurons[-1]])
    if xlabel:
        plt.xlabel('Number of Neurons')
    plt.ylabel('Step Time (ms)')
    plt.yscale('log')
    plt.xscale('log')
    plt.title(title, loc='left', weight='bold')
    plt.axhline(y=rt, color='black', linestyle='--', label='Real-Time')
    if legend:
        plt.legend()

def figure_backends(size, colors):
    dataNonspikingDense = pickle.load(open('../benchmarking/backendSpeed/dataBackendTimesNonspikingDense.p', 'rb'))

    dataNonspikingSparse = pickle.load(open('../benchmarking/backendSpeed/dataBackendTimesNonspikingSparse.p', 'rb'))

    dataSpikingDense = pickle.load(open('../benchmarking/backendSpeed/dataBackendTimesSpikingDense.p', 'rb'))

    dataSpikingSparse = pickle.load(open('../benchmarking/backendSpeed/dataBackendTimes.p', 'rb'))

    neurons = dataNonspikingDense['shape']
    sns_raw_times_nonspiking_dense = [dataNonspikingDense['numpy'], dataNonspikingDense['torchCPU'],
                                      dataNonspikingDense['torchGPU'],
                                      dataNonspikingDense['manual']]

    sns_raw_times_nonspiking_sparse = [dataNonspikingSparse['numpy'], dataNonspikingSparse['torchCPU'],
                                       dataNonspikingSparse['torchGPU'],
                                       dataNonspikingSparse['manual'], dataNonspikingSparse['sparseCPU'],
                                       dataNonspikingSparse['sparseGPU']]

    sns_raw_times_spiking_dense = [dataSpikingDense['numpy'], dataSpikingDense['torchCPU'],
                                   dataSpikingDense['torchGPU'],
                                   dataSpikingDense['manual']]

    sns_raw_times_spiking_sparse = [dataSpikingSparse['numpy'], dataSpikingSparse['torchCPU'],
                                    dataSpikingSparse['torchGPU'],
                                    dataSpikingSparse['manual'], dataSpikingSparse['sparseCPU'],
                                    dataSpikingSparse['sparseGPU']]

    labels_sns = ['Numpy', 'Torch (CPU)', 'Torch (GPU)', 'Iterative', 'Sparse (CPU)', 'Sparse (GPU)']
    linestyles_sns = ['solid', '--', ':', 'solid', '--', ':']

    plt.figure(figsize=size, constrained_layout=True)
    plt.subplot(2, 2, 1)
    plot_multiple_shaded(neurons, 'A', sns_raw_times_nonspiking_dense, colors, labels_sns,
                         linestyles_sns,xlabel=False,rt=1.0, legend=True)
    plt.title('Dense Non-Spiking')
    plt.subplot(2, 2, 2)
    plot_multiple_shaded(neurons, 'B', sns_raw_times_nonspiking_sparse, colors, labels_sns,
                         linestyles_sns, xlabel=False,rt=1.0)
    plt.title('Sparse Non-Spiking')
    plt.subplot(2, 2, 3)
    plot_multiple_shaded(neurons, 'C', sns_raw_times_spiking_dense, colors, labels_sns,
                         linestyles_sns,rt=0.1)
    plt.title('Dense Spiking')
    plt.subplot(2, 2, 4)
    plot_multiple_shaded(neurons, 'D', sns_raw_times_spiking_sparse, colors, labels_sns,
                         linestyles_sns,rt=0.1)
    plt.title('Sparse Spiking')
    # plt.tight_layout()
    plt.savefig('figure_backends.pdf')
    plt.savefig('figure_backends.svg')

def figure_software(size, colors):
    dataNonspikingDense = pickle.load(open('../benchmarking/backendSpeed/dataBackendTimesNonspikingDense.p', 'rb'))
    dataBrianNonspikingDense = pickle.load(open('../benchmarking/backendSpeed/dataBrianTimesNonspikingDense.p', 'rb'))
    dataBrianCUDANonspikingDense = pickle.load(
        open('../benchmarking/backendSpeed/dataBrianCudaTimesNonspikingDense.p', 'rb'))
    dataNengoNonspikingDense = pickle.load(open('../benchmarking/backendSpeed/dataNengoTimesNonspikingDense.p', 'rb'))
    dataANNarchyNonspikingDense = pickle.load(
        open('../benchmarking/otherSoftware/annarchy/dataANNarchyTimesNonspikingDense.p', 'rb'))
    dataANNarchyCUDANonspikingDense = pickle.load(
        open('../benchmarking/otherSoftware/annarchy/dataANNarchyCUDATimesNonspikingDense.p', 'rb'))

    dataNonspikingSparse = pickle.load(open('../benchmarking/backendSpeed/dataBackendTimesNonspikingSparse.p', 'rb'))
    dataBrianNonspikingSparse = pickle.load(open('../benchmarking/backendSpeed/dataBrianTimesNonspikingSparse.p', 'rb'))
    dataBrianCUDANonspikingSparse = pickle.load(
        open('../benchmarking/backendSpeed/dataBrianCudaTimesNonspikingSparse.p', 'rb'))
    dataNengoNonspikingSparse = pickle.load(open('../benchmarking/backendSpeed/dataNengoTimesNonspikingSparse.p', 'rb'))
    dataANNarchyNonspikingSparse = pickle.load(
        open('../benchmarking/otherSoftware/annarchy/dataANNarchyTimesNonspikingSparse.p', 'rb'))
    dataANNarchyCUDANonspikingSparse = pickle.load(
        open('../benchmarking/otherSoftware/annarchy/dataANNarchyCUDATimesNonspikingSparse.p', 'rb'))

    dataSpikingDense = pickle.load(open('../benchmarking/backendSpeed/dataBackendTimesSpikingDense.p', 'rb'))
    dataBrianSpikingDense = pickle.load(open('../benchmarking/backendSpeed/dataBrianTimesSpikingDense.p', 'rb'))
    dataBrianCUDASpikingDense = pickle.load(open('../benchmarking/backendSpeed/dataBrianCudaTimesSpikingDense.p', 'rb'))
    dataNengoSpikingDense = pickle.load(open('../benchmarking/backendSpeed/dataNengoTimesSpikingDense.p', 'rb'))
    dataANNarchySpikingDense = pickle.load(
        open('../benchmarking/otherSoftware/annarchy/dataANNarchyTimesSpikingDense.p', 'rb'))
    dataANNarchyCUDASpikingDense = pickle.load(
        open('../benchmarking/otherSoftware/annarchy/dataANNarchyCUDATimesSpikingDense.p', 'rb'))

    dataSpikingSparse = pickle.load(open('../benchmarking/backendSpeed/dataBackendTimes.p', 'rb'))
    dataBrianSpikingSparse = pickle.load(open('../benchmarking/backendSpeed/dataBrianTimesSpikingSparse.p', 'rb'))
    dataBrianCUDASpikingSparse = pickle.load(
        open('../benchmarking/backendSpeed/dataBrianCudaTimesSpikingSparse.p', 'rb'))
    dataNengoSpikingSparse = pickle.load(open('../benchmarking/backendSpeed/dataNengoTimesSpikingSparse.p', 'rb'))
    dataANNarchySpikingSparse = pickle.load(
        open('../benchmarking/otherSoftware/annarchy/dataANNarchyTimesSpikingSparse.p', 'rb'))
    dataANNarchyCUDASpikingSparse = pickle.load(
        open('../benchmarking/otherSoftware/annarchy/dataANNarchyCUDATimesSpikingSparse.p', 'rb'))

    neurons = dataNonspikingDense['shape']
    sns_raw_times_nonspiking_dense = [dataNonspikingDense['numpy'], dataNonspikingDense['torchCPU'],
                                      dataNonspikingDense['torchGPU'],
                                      dataNonspikingDense['manual']]

    sns_raw_times_nonspiking_sparse = [dataNonspikingSparse['numpy'], dataNonspikingSparse['torchCPU'],
                                       dataNonspikingSparse['torchGPU'],
                                       dataNonspikingSparse['manual'], dataNonspikingSparse['sparseCPU'],
                                       dataNonspikingSparse['sparseGPU']]

    sns_raw_times_spiking_dense = [dataSpikingDense['numpy'], dataSpikingDense['torchCPU'],
                                   dataSpikingDense['torchGPU'],
                                   dataSpikingDense['manual']]

    sns_raw_times_spiking_sparse = [dataSpikingSparse['numpy'], dataSpikingSparse['torchCPU'],
                                    dataSpikingSparse['torchGPU'],
                                    dataSpikingSparse['manual'], dataSpikingSparse['sparseCPU'],
                                    dataSpikingSparse['sparseGPU']]
    best_sns_nonspiking_dense = get_best(neurons, sns_raw_times_nonspiking_dense)
    best_sns_nonspiking_sparse = get_best(neurons, sns_raw_times_nonspiking_sparse)
    best_sns_spiking_dense = get_best(neurons, sns_raw_times_spiking_dense)
    best_sns_spiking_sparse = get_best(neurons, sns_raw_times_spiking_sparse)

    best_brian_nonspiking_dense = get_best(neurons,
                                           [dataBrianNonspikingDense['brian'], dataBrianCUDANonspikingDense['brian']])
    best_brian_nonspiking_sparse = get_best(neurons, [dataBrianNonspikingSparse['brian'],
                                                      dataBrianCUDANonspikingSparse['brian']])
    best_brian_spiking_dense = get_best(neurons, [dataBrianSpikingDense['brian'], dataBrianCUDASpikingDense['brian']])
    best_brian_spiking_sparse = dataBrianSpikingSparse['brian']

    best_annarchy_nonspiking_dense = set_zero_to_nan_2d(
        get_best(neurons, [dataANNarchyNonspikingDense['annarchy'], dataANNarchyCUDANonspikingDense['annarchy']]))
    best_annarchy_nonspiking_sparse = get_best(neurons, [dataANNarchyNonspikingSparse['annarchy'],
                                                         dataANNarchyCUDANonspikingSparse['annarchy']])
    best_annarchy_spiking_dense = get_best(neurons, [dataANNarchySpikingDense['annarchy'],
                                                     dataANNarchyCUDASpikingDense['annarchy']])
    best_annarchy_spiking_sparse = get_best(neurons, [dataANNarchySpikingSparse['annarchy'],
                                                      dataANNarchyCUDASpikingSparse['annarchy']])

    labels_best = ['SNS-Toolbox', 'Brian2', 'Nengo', 'ANNarchy']
    linestyles_best = ['solid', 'dotted', 'dashed', 'dashdot']

    plt.figure(figsize=size, constrained_layout=True)
    plt.subplot(2, 2, 1)
    plot_multiple_shaded(neurons, 'A',
                         [best_sns_nonspiking_dense, best_brian_nonspiking_dense, dataNengoNonspikingDense['nengo'],
                          best_annarchy_nonspiking_dense], colors, labels_best, linestyles_best,xlabel=False,rt=1.0)
    plt.title('Dense Non-Spiking')
    plt.subplot(2, 2, 2)
    plot_multiple_shaded(neurons, 'B',
                         [best_sns_nonspiking_sparse, best_brian_nonspiking_sparse, dataNengoNonspikingSparse['nengo'],
                          best_annarchy_nonspiking_sparse], colors, labels_best, linestyles_best, xlabel=False,rt=1.0)
    plt.title('Sparse Non-Spiking')
    plt.subplot(2, 2, 3)
    plot_multiple_shaded(neurons, 'C',
                         [best_sns_spiking_dense, best_brian_spiking_dense, dataNengoSpikingDense['nengo'],
                          best_annarchy_spiking_dense], colors, labels_best, linestyles_best,rt=0.1)
    plt.title('Dense Spiking')
    plt.subplot(2, 2, 4)
    plot_multiple_shaded(neurons, 'D',
                         [best_sns_spiking_sparse, best_brian_spiking_sparse, dataNengoSpikingSparse['nengo'],
                          best_annarchy_spiking_sparse], colors, labels_best, linestyles_best,rt=0.1, legend=True)
    plt.title('Sparse Spiking')
    # plt.tight_layout()
    plt.savefig('figure_software.pdf')
    plt.savefig('figure_software.svg')

def figure_hardware(size, colors):
    dataNUCNonspikingDense = pickle.load(open('../benchmarking/otherHardware/intel_nuc/dataNUCNonspikingDense.p', 'rb'))
    dataNanoNonspikingDense = pickle.load(
        open('../benchmarking/otherHardware/jetson_nano/dataJetsonNonspikingDense.p', 'rb'))
    dataRpiNonspikingDense = pickle.load(open('../benchmarking/otherHardware/rpi_3b/dataRpi3bNonspikingDense.p', 'rb'))
    dataNUCNonspikingSparse = pickle.load(
        open('../benchmarking/otherHardware/intel_nuc/dataNUCNonspikingSparse.p', 'rb'))
    dataNanoNonspikingSparse = pickle.load(
        open('../benchmarking/otherHardware/jetson_nano/dataJetsonNanoNonspikingSparse.p', 'rb'))
    dataRpiNonspikingSparse = pickle.load(
        open('../benchmarking/otherHardware/rpi_3b/dataRpi3bNonspikingSparse.p', 'rb'))
    dataNUCSpikingDense = pickle.load(open('../benchmarking/otherHardware/intel_nuc/dataNUCSpikingDense.p', 'rb'))
    dataNanoSpikingDense = pickle.load(
        open('../benchmarking/otherHardware/jetson_nano/dataJetsonNanoSpikingDense.p', 'rb'))
    dataRpiSpikingDense = pickle.load(open('../benchmarking/otherHardware/rpi_3b/dataRpi3bSpikingDense.p', 'rb'))
    dataNUCSpikingSparse = pickle.load(open('../benchmarking/otherHardware/intel_nuc/dataNUCSpikingSparse.p', 'rb'))
    dataNanoSpikingSparse = pickle.load(
        open('../benchmarking/otherHardware/jetson_nano/dataJetsonNanoSpikingSparse.p', 'rb'))
    dataRpiSpikingSparse = pickle.load(open('../benchmarking/otherHardware/rpi_3b/dataRpi3bSpikingSparse.p', 'rb'))

    raw_nuc_nonspiking_dense = [dataNUCNonspikingDense['numpy'], dataNUCNonspikingDense['torchCPU'],
                                dataNUCNonspikingDense['sparseCPU'], dataNUCNonspikingDense['manual'],
                                dataNUCNonspikingDense['torchGPU'], dataNUCNonspikingDense['sparseGPU']]
    raw_nano_nonspiking_dense = [dataNanoNonspikingDense['numpy'], dataNanoNonspikingDense['torchCPU'],
                                 dataNanoNonspikingDense['sparseCPU'], dataNanoNonspikingDense['manual'],
                                 dataNanoNonspikingDense['torchGPU'], dataNanoNonspikingDense['sparseGPU']]
    raw_rpi_nonspiking_dense = [dataRpiNonspikingDense['numpy'], dataRpiNonspikingDense['torchCPU'],
                                dataRpiNonspikingDense['sparseCPU'], dataRpiNonspikingDense['manual']]
    raw_nuc_nonspiking_sparse = [dataNUCNonspikingSparse['numpy'], dataNUCNonspikingSparse['torchCPU'],
                                 dataNUCNonspikingSparse['sparseCPU'], dataNUCNonspikingSparse['manual'],
                                 dataNUCNonspikingSparse['torchGPU'], dataNUCNonspikingSparse['sparseGPU']]
    raw_nano_nonspiking_sparse = [dataNanoNonspikingSparse['numpy'], dataNanoNonspikingSparse['torchCPU'],
                                  dataNanoNonspikingSparse['sparseCPU'], dataNanoNonspikingSparse['manual'],
                                  dataNanoNonspikingSparse['torchGPU'], dataNanoNonspikingSparse['sparseGPU']]
    raw_rpi_nonspiking_sparse = [dataRpiNonspikingSparse['numpy'], dataRpiNonspikingSparse['torchCPU'],
                                 dataRpiNonspikingSparse['sparseCPU'], dataRpiNonspikingSparse['manual']]
    raw_nuc_spiking_dense = [dataNUCSpikingDense['numpy'], dataNUCSpikingDense['torchCPU'],
                             dataNUCSpikingDense['sparseCPU'], dataNUCSpikingDense['manual'],
                             dataNUCSpikingDense['torchGPU'], dataNUCSpikingDense['sparseGPU']]
    raw_nano_spiking_dense = [dataNanoSpikingDense['numpy'], dataNanoSpikingDense['torchCPU'],
                              dataNanoSpikingDense['sparseCPU'], dataNanoSpikingDense['manual'],
                              dataNanoSpikingDense['torchGPU'], dataNanoSpikingDense['sparseGPU']]
    raw_rpi_spiking_dense = [dataRpiSpikingDense['numpy'], dataRpiSpikingDense['torchCPU'],
                             dataRpiSpikingDense['sparseCPU'], dataRpiSpikingDense['manual']]
    raw_nuc_spiking_sparse = [dataNUCSpikingSparse['numpy'], dataNUCSpikingSparse['torchCPU'],
                              dataNUCSpikingSparse['sparseCPU'], dataNUCSpikingSparse['manual'],
                              dataNUCSpikingSparse['torchGPU'], dataNUCSpikingSparse['sparseGPU']]
    raw_nano_spiking_sparse = [dataNanoSpikingSparse['numpy'], dataNanoSpikingSparse['torchCPU'],
                               dataNanoSpikingSparse['sparseCPU'], dataNanoSpikingSparse['manual'],
                               dataNanoSpikingSparse['torchGPU'], dataNanoSpikingSparse['sparseGPU']]
    raw_rpi_spiking_sparse = [dataRpiSpikingSparse['numpy'], dataRpiSpikingSparse['torchCPU'],
                              dataRpiSpikingSparse['sparseCPU'], dataRpiSpikingSparse['manual']]

    neurons_hardware = dataRpiNonspikingDense['shape']

    best_nuc_nonspiking_dense = get_best(neurons_hardware, raw_nuc_nonspiking_dense)
    best_nano_nonspiking_dense = get_best(neurons_hardware, raw_nano_nonspiking_dense)
    best_rpi_nonspiking_dense = set_zero_to_nan_2d(get_best(neurons_hardware, raw_rpi_nonspiking_dense))
    best_nuc_nonspiking_sparse = get_best(neurons_hardware, raw_nuc_nonspiking_sparse)
    best_nano_nonspiking_sparse = get_best(neurons_hardware, raw_nano_nonspiking_sparse)
    best_rpi_nonspiking_sparse = set_zero_to_nan_2d(get_best(neurons_hardware, raw_rpi_nonspiking_sparse))
    best_nuc_spiking_dense = get_best(neurons_hardware, raw_nuc_spiking_dense)
    best_nano_spiking_dense = get_best(neurons_hardware, raw_nano_spiking_dense)
    best_rpi_spiking_dense = set_zero_to_nan_2d(get_best(neurons_hardware, raw_rpi_spiking_dense))
    best_nuc_spiking_sparse = get_best(neurons_hardware, raw_nuc_spiking_sparse)
    best_nano_spiking_sparse = get_best(neurons_hardware, raw_nano_spiking_sparse)
    best_rpi_spiking_sparse = set_zero_to_nan_2d(get_best(neurons_hardware, raw_rpi_spiking_sparse))

    labels_hardware = ['NUC', 'Jetson Nano', 'Rpi 3B']
    linestyles_hardware = ['solid', 'dotted', 'dashed']

    plt.figure(figsize=size, constrained_layout=True)
    plt.subplot(2, 2, 1)
    plot_multiple_shaded(neurons_hardware, 'A',
                         [best_nuc_nonspiking_dense, best_nano_nonspiking_dense, best_rpi_nonspiking_dense], colors,
                         labels_hardware, linestyles_hardware, xlabel=False,rt=1.0)
    plt.title('Dense Non-Spiking')
    plt.subplot(2, 2, 2)
    plot_multiple_shaded(neurons_hardware, 'B',
                         [best_nuc_nonspiking_sparse, best_nano_nonspiking_sparse, best_rpi_nonspiking_sparse], colors,
                         labels_hardware, linestyles_hardware, legend=True, xlabel=False,rt=1.0)
    plt.title('Sparse Non-Spiking')
    plt.subplot(2, 2, 3)
    plot_multiple_shaded(neurons_hardware, 'C',
                         [best_nuc_spiking_dense, best_nano_spiking_dense, best_rpi_spiking_dense], colors,
                         labels_hardware, linestyles_hardware,rt=0.1)
    plt.title('Dense Spiking')
    plt.subplot(2, 2, 4)
    plot_multiple_shaded(neurons_hardware, 'D',
                         [best_nuc_spiking_sparse, best_nano_spiking_sparse, best_rpi_spiking_sparse], colors,
                         labels_hardware, linestyles_hardware,rt=0.1)
    plt.title('Sparse Spiking')

    # plt.tight_layout()
    plt.savefig('figure_hardware.pdf')
    plt.savefig('figure_hardware.svg')

def figure_hindlimb(size, colors):
    data = pd.read_csv('leg_outputs.csv')
    fig = plt.figure(figsize=size, constrained_layout=True)
    gs = fig.add_gridspec(nrows=3, ncols=7)
    plt.subplot(gs[:,:3])
    plt.imshow(np.asarray(Image.open('sns_diagram.png')), aspect='auto')
    plt.title('A', loc='left', weight='bold')
    plt.axis('off')

    plt.subplot(gs[1:,3])
    leg = np.asarray(Image.open('Capture0.PNG'))
    plt.imshow(leg, aspect='auto')
    plt.title('C', loc='left', weight='bold')
    plt.axis('off')

    plt.subplot(gs[0,3])
    potentials = np.arange(-110, -30, 0.1)
    activation = np.zeros(len(potentials))
    for i in range(len(potentials)):
        activation[i] = stim2activation(potentials[i])
    plt.plot(potentials, activation, color=colors[9], label='Sigmoid Clipped from 0-1')
    plt.ylabel('Muscle Activation')
    plt.xlabel('Neuron Potential (mV)')
    plt.title('B', loc='left', weight='bold')
    plt.title('Activation')
    # plt.text(-105, 0.7, r"$1/(1+e^{0.1532(-70-Potential)}) - 0.01$")
    # plt.text(-101, 0.78, 'Clipped between 0 and 1')

    plt.subplot(gs[0,4])
    plt.plot(data['Time'].to_numpy()[-10000:], data['L_RG_Flx'].to_numpy()[-10000:], label='HC_ext', color=colors[0])
    plt.plot(data['Time'].to_numpy()[-10000:], data['L_RG_Ext'].to_numpy()[-10000:], label='HC_flx', color=colors[1])
    # plt.xlabel('Time (ms)')
    plt.ylabel('Potential (mV)')
    plt.title('RG')
    plt.title('D', loc='left', weight='bold')
    plt.legend()

    plt.subplot(gs[0,5])
    plt.plot(data['Time'].to_numpy()[-10000:], data['L_PF_Hip_Flx'].to_numpy()[-10000:], label='HC_ext', color=colors[2])
    plt.plot(data['Time'].to_numpy()[-10000:], data['L_PF_Hip_Ext'].to_numpy()[-10000:], label='HC_flx', color=colors[3])
    # plt.xlabel('Time (ms)')
    plt.ylabel('Potential (mV)')
    plt.title('Hip PF')
    plt.title('E', loc='left', weight='bold')
    plt.legend()

    plt.subplot(gs[0,6])
    plt.plot(data['Time'].to_numpy()[-10000:], data['L_PF_KA_Flx'].to_numpy()[-10000:], label='HC_ext', color=colors[2])
    plt.plot(data['Time'].to_numpy()[-10000:], data['L_PF_KA_Ext'].to_numpy()[-10000:], label='HC_flx', color=colors[3])
    # plt.xlabel('Time (ms)')
    plt.ylabel('Potential (mV)')
    plt.title('Knee/Ankle PF')
    plt.title('F', loc='left', weight='bold')
    plt.legend()

    plt.subplot(gs[1,4])
    plt.plot(data['Time'].to_numpy()[-10000:], data['L_MN_Hip_Ext'].to_numpy()[-10000:], label='Extensor', color=colors[4])
    plt.plot(data['Time'].to_numpy()[-10000:], data['L_MN_Hip_Flx'].to_numpy()[-10000:], label='Flexor', color=colors[5])
    # plt.xlabel('Time (ms)')
    plt.ylabel('Potential (mV)')
    plt.title('Hip MN')
    plt.title('G', loc='left', weight='bold')
    plt.legend()

    plt.subplot(gs[1,5])
    plt.plot(data['Time'].to_numpy()[-10000:], data['L_MN_Knee_Ext'].to_numpy()[-10000:], label='Extensor', color=colors[4])
    plt.plot(data['Time'].to_numpy()[-10000:], data['L_MN_Knee_Flx'].to_numpy()[-10000:], label='Flexor', color=colors[5])
    # plt.xlabel('Time (ms)')
    plt.ylabel('Potential (mV)')
    plt.title('Knee MN')
    plt.title('H', loc='left', weight='bold')
    plt.legend()

    plt.subplot(gs[1,6])
    plt.plot(data['Time'].to_numpy()[-10000:], data['L_MN_Ankle_Ext'].to_numpy()[-10000:], label='Extensor', color=colors[4])
    plt.plot(data['Time'].to_numpy()[-10000:], data['L_MN_Ankle_Flx'].to_numpy()[-10000:], label='Flexor', color=colors[5])
    # plt.xlabel('Time (ms)')
    plt.ylabel('Potential (mV)')
    plt.title('Ankle MN')
    plt.title('I', loc='left', weight='bold')
    plt.legend()

    plt.subplot(gs[2, 4])
    plt.plot(data['Time'].to_numpy()[-10000:], data['L_Hip_Joint_pos'].to_numpy()[-10000:], label='Left', color=colors[7])
    plt.xlabel('Time (ms)')
    plt.ylabel('Angle (rad)')
    plt.title('Hip Angle')
    plt.title('J', loc='left', weight='bold')

    plt.subplot(gs[2, 5])
    plt.plot(data['Time'].to_numpy()[-10000:], data['L_Knee_Joint_pos'].to_numpy()[-10000:], label='Left', color=colors[7])
    plt.xlabel('Time (ms)')
    plt.ylabel('Angle (rad)')
    plt.title('Knee Angle')
    plt.title('K', loc='left', weight='bold')

    plt.subplot(gs[2, 6])
    plt.plot(data['Time'].to_numpy()[-10000:], data['L_Ankle_Joint_pos'].to_numpy()[-10000:], label='Left', color=colors[7])
    plt.xlabel('Time (ms)')
    plt.ylabel('Angle (rad)')
    plt.title('Ankle Angle')
    plt.title('L', loc='left', weight='bold')

    plt.savefig('figure_hindlimb.pdf')
    plt.savefig('figure_hindlimb.svg')

def figure_hindlimb_right(size, colors):
    data = pd.read_csv('leg_outputs.csv')
    fig = plt.figure(figsize=size, constrained_layout=True)
    gs = fig.add_gridspec(nrows=3, ncols=3)


    plt.subplot(gs[0,0])
    plt.plot(data['Time'].to_numpy()[-10000:], data['R_RG_Flx'].to_numpy()[-10000:], label='HC_ext', color=colors[0])
    plt.plot(data['Time'].to_numpy()[-10000:], data['R_RG_Ext'].to_numpy()[-10000:], label='HC_flx', color=colors[1])
    # plt.xlabel('Time (ms)')
    plt.ylabel('Potential (mV)')
    plt.title('RG')
    plt.title('A', loc='left', weight='bold')
    plt.legend()

    plt.subplot(gs[0,1])
    plt.plot(data['Time'].to_numpy()[-10000:], data['R_PF_Hip_Flx'].to_numpy()[-10000:], label='HC_ext', color=colors[2])
    plt.plot(data['Time'].to_numpy()[-10000:], data['R_PF_Hip_Ext'].to_numpy()[-10000:], label='HC_flx', color=colors[3])
    # plt.xlabel('Time (ms)')
    plt.ylabel('Potential (mV)')
    plt.title('Hip PF')
    plt.title('B', loc='left', weight='bold')
    plt.legend()

    plt.subplot(gs[0,2])
    plt.plot(data['Time'].to_numpy()[-10000:], data['R_PF_KA_Flx'].to_numpy()[-10000:], label='HC_ext', color=colors[2])
    plt.plot(data['Time'].to_numpy()[-10000:], data['R_PF_KA_Ext'].to_numpy()[-10000:], label='HC_flx', color=colors[3])
    # plt.xlabel('Time (ms)')
    plt.ylabel('Potential (mV)')
    plt.title('Knee/Ankle PF')
    plt.title('C', loc='left', weight='bold')
    plt.legend()

    plt.subplot(gs[1,0])
    plt.plot(data['Time'].to_numpy()[-10000:], data['R_MN_Hip_Ext'].to_numpy()[-10000:], label='Extensor', color=colors[4])
    plt.plot(data['Time'].to_numpy()[-10000:], data['R_MN_Hip_Flx'].to_numpy()[-10000:], label='Flexor', color=colors[5])
    # plt.xlabel('Time (ms)')
    plt.ylabel('Potential (mV)')
    plt.title('Hip MN')
    plt.title('D', loc='left', weight='bold')
    plt.legend()

    plt.subplot(gs[1,1])
    plt.plot(data['Time'].to_numpy()[-10000:], data['R_MN_Knee_Ext'].to_numpy()[-10000:], label='Extensor', color=colors[4])
    plt.plot(data['Time'].to_numpy()[-10000:], data['R_MN_Knee_Flx'].to_numpy()[-10000:], label='Flexor', color=colors[5])
    # plt.xlabel('Time (ms)')
    plt.ylabel('Potential (mV)')
    plt.title('Knee MN')
    plt.title('E', loc='left', weight='bold')
    plt.legend()

    plt.subplot(gs[1,2])
    plt.plot(data['Time'].to_numpy()[-10000:], data['R_MN_Ankle_Ext'].to_numpy()[-10000:], label='Extensor', color=colors[4])
    plt.plot(data['Time'].to_numpy()[-10000:], data['R_MN_Ankle_Flx'].to_numpy()[-10000:], label='Flexor', color=colors[5])
    # plt.xlabel('Time (ms)')
    plt.ylabel('Potential (mV)')
    plt.title('Ankle MN')
    plt.title('F', loc='left', weight='bold')
    plt.legend()

    plt.subplot(gs[2, 0])
    plt.plot(data['Time'].to_numpy()[-10000:], data['R_Hip_Joint_pos'].to_numpy()[-10000:], label='Left', color=colors[7])
    plt.xlabel('Time (ms)')
    plt.ylabel('Angle (rad)')
    plt.title('Hip Angle')
    plt.title('G', loc='left', weight='bold')

    plt.subplot(gs[2, 1])
    plt.plot(data['Time'].to_numpy()[-10000:], data['R_Knee_Joint_pos'].to_numpy()[-10000:], label='Left', color=colors[7])
    plt.xlabel('Time (ms)')
    plt.ylabel('Angle (rad)')
    plt.title('Knee Angle')
    plt.title('H', loc='left', weight='bold')

    plt.subplot(gs[2, 2])
    plt.plot(data['Time'].to_numpy()[-10000:], data['R_Ankle_Joint_pos'].to_numpy()[-10000:], label='Left', color=colors[7])
    plt.xlabel('Time (ms)')
    plt.ylabel('Angle (rad)')
    plt.title('Ankle Angle')
    plt.title('I', loc='left', weight='bold')

    plt.savefig('figure_hindlimb_right.pdf')
    plt.savefig('figure_hindlimb_right.svg')

def figure_angle_comparison(size, colors):
    data = pd.read_csv('leg_outputs.csv')
    animatlab_data = pd.read_csv('animatlab_L_joint_angles.txt', delimiter='\t')
    plt.figure(figsize=size, constrained_layout=True)

    plt.subplot(1,3,1)
    plt.plot(data['Time'].to_numpy()[-10000:], data['L_Hip_Joint_pos'].to_numpy()[-10000:], label='SNS-Toolbox',
             color=colors[7],linestyle='--')
    plt.plot(animatlab_data['Time'].to_numpy()[-20000:] * 1000, animatlab_data['LH_HipZ'].to_numpy()[-20000:], label='AnimatLab', color='C0')
    plt.xlabel('Time (ms)')
    plt.ylabel('Angle (rad)')
    plt.title('Hip Angle')
    plt.title('A', loc='left', weight='bold')
    plt.legend()
    print(len(animatlab_data['Time'].to_numpy()))
    print(len(data['Time'].to_numpy()))

    plt.subplot(1,3,2)
    plt.plot(data['Time'].to_numpy()[-10000:], data['L_Knee_Joint_pos'].to_numpy()[-10000:], label='SNS-Toolbox',
             color=colors[7], linestyle='--')
    plt.plot(animatlab_data['Time'].to_numpy()[-20000:]*1000, animatlab_data['LH_Knee'].to_numpy()[-20000:],label='AnimatLab',color='C0')
    plt.xlabel('Time (ms)')
    plt.ylabel('Angle (rad)')
    plt.title('Knee Angle')
    plt.title('B', loc='left', weight='bold')
    plt.legend()

    plt.subplot(1,3,3)
    plt.plot(data['Time'].to_numpy()[-10000:], data['L_Ankle_Joint_pos'].to_numpy()[-10000:], label='SNS-Toolbox',
             color=colors[7], linestyle='--')
    plt.plot(animatlab_data['Time'].to_numpy()[-20000:]*1000, animatlab_data['LH_AnkleZ'].to_numpy()[-20000:],label='AnimatLab',color='C0')
    plt.xlabel('Time (ms)')
    plt.ylabel('Angle (rad)')
    plt.title('Ankle Angle')
    plt.title('C', loc='left', weight='bold')
    plt.legend()

    plt.savefig('figure_angle_comparison.pdf')
    plt.savefig('figure_angle_comparison.svg')

def figure_ros_neurons(size,colors):
    df = pd.read_csv('../ros/sns_ros_demo_neurons.csv')
    ccw = df['field.data0']
    cw = df['field.data1']
    speed = df['field.data2']

    plt.figure(figsize=size,constrained_layout=True)
    plt.plot(ccw,color=colors[0], label='CCW')
    plt.plot(cw, color=colors[1], label='CW', linestyle='--')
    plt.plot(speed, color=colors[2], label='Speed', linestyle=':')
    plt.xlabel('t (ms)')
    plt.ylabel('Neural Potential (mV)')
    plt.title('C', loc='left', weight='bold')
    plt.title('Neural Activity')
    plt.legend()
    plt.savefig('figure_ros_neurons.svg')

def figure_ros_trajectory():
    df = pd.read_csv('../ros/sns_ros_demo_pose.csv')
    x = df['field.position.x'].to_numpy()
    y = df['field.position.y'].to_numpy()
    plt.figure()
    plt.plot(x, y, linestyle='--')
    plt.xlim([-50, 50])
    plt.ylim([-50, 50])
    plt.gca().set_aspect('equal')
    plt.savefig('figure_ros_trajectory.svg')

def main():
    sea.set_theme(palette='colorblind', style='ticks')
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10']

    figure_backends((10, 6), colors)
    figure_software((10, 6), colors)
    figure_hardware((10, 6), colors)
    # figure_hindlimb((15,10), colors)
    # figure_hindlimb_right((10,8), colors)
    # figure_angle_comparison((10,3), colors)
    # figure_ros_neurons((10,3),colors)
    # figure_ros_trajectory()

    plt.show()

if __name__ == '__main__':
    main()
