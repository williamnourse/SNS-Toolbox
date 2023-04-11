import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import sns_toolbox.backends
from sklearn.decomposition import PCA
import scipy

from sns_toolbox.networks import Network
from sns_toolbox.neurons import SpikingNeuron
from sns_toolbox.connections import SpikingMatrixConnection
from sns_toolbox.plot_utilities import spike_raster_plot

def gen_network(num_neurons, ratio_conn, g_max_ex, g_max_in, reversal_ex, reversal_in, g_m, c_m, rest, activity_range, dt, rng, backend, device,tau):
    g_max_matrix = np.zeros([num_neurons, num_neurons])
    reversal_matrix = np.zeros_like(g_max_matrix)
    time_constant_matrix = np.zeros_like(g_max_matrix)+tau
    transmission_delay = np.zeros_like(g_max_matrix)

    for i in range(num_neurons):
        pre_inds_ex = rng.choice(int(num_neurons / 2), size=int(ratio_conn * num_neurons / 2), replace=False)
        pre_inds_in = rng.choice(int(num_neurons / 2), size=int(ratio_conn * num_neurons / 2), replace=False) + int(num_neurons / 2)
        g_max_matrix[i,pre_inds_ex] = g_max_ex
        g_max_matrix[i,pre_inds_in] = g_max_in
        reversal_matrix[i, pre_inds_ex] = reversal_ex
        reversal_matrix[i, pre_inds_in] = reversal_in

    net = Network('Balanced Sequence Generator')

    neuron_type = SpikingNeuron(membrane_conductance=g_m, membrane_capacitance=c_m, resting_potential=rest, threshold_proportionality_constant=0.0, threshold_initial_value=1.0)

    net.add_population(neuron_type, shape=[num_neurons], name='BSG', initial_value=rng.random(num_neurons))

    connection = SpikingMatrixConnection(g_max_matrix, reversal_matrix, time_constant=time_constant_matrix,transmission_delay=transmission_delay)
    net.add_connection(connection,'BSG', 'BSG')

    net.add_input('BSG')
    net.add_output('BSG',spiking=True)

    model = net.compile(dt=dt, backend=backend, device=device)

    return model

def calc_eigenvalues(model):
    eigenvalues, _ = np.linalg.eig(model.g_max_non*model.del_e)
    eigenvalues_abs = np.abs(eigenvalues)
    max_eig_magnitude = np.max(eigenvalues_abs)
    max_eig_index = np.argmax(eigenvalues_abs)
    max_eig = eigenvalues[max_eig_index]

    return eigenvalues, max_eig, max_eig_index, max_eig_magnitude

def plot_eigenvalues(eigenvalues, max_eig):
    fig = plt.figure()
    plt.title('Eigenvalues at Rest')
    plt.scatter(eigenvalues.real,eigenvalues.imag, label='Eigenvalues')

    plt.scatter(max_eig.real,max_eig.imag,color='red', label='Maximum Eigenvalue')
    circ = plt.Circle((0,0),1, color='black',fill=False,ls='--', label='Unit Circle')
    plt.axhline(y=0,color='black')
    plt.axvline(x=0,color='black')
    ax = plt.gca()
    ax.add_patch(circ)
    ax.set_aspect('equal')
    plt.legend()

    return fig

def plot_connectivity(model):
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["darkblue", "blue", "white", "red", "darkred"])
    norm = plt.Normalize(-1, 1)

    fig = plt.figure()
    plt.subplot(1,2,1)
    plt.title('G_max')
    plt.imshow(model.g_max_spike, cmap=cmap, norm=norm)
    # plt.colorbar()
    plt.xlabel('Presynaptic Index')
    plt.ylabel('Postsynaptic Index')

    plt.subplot(1,2,2)
    plt.title('E_syn')
    plt.imshow(model.del_e,cmap=cmap)
    plt.colorbar()
    plt.xlabel('Presynaptic Index')
    plt.ylabel('Postsynaptic Index')

    return fig


def run_network(model: sns_toolbox.backends.Backend, num_steps, stim, dt):
    t = np.arange(0, num_steps * dt, dt)
    data = np.zeros([num_steps, model.num_neurons])
    model.reset()
    for i in range(num_steps):
        data[i,:] = model([stim])
    return data, t

def plot_activity(data, t, title, num_neurons):
    data = data.transpose()
    fig = plt.figure()
    plt.subplot(2, 1, 1)
    plt.title(title)
    spike_raster_plot(t,data[:][:])
    plt.ylabel('Response (mV)')
    plt.xlim([t[0], t[-1]])

    plt.subplot(2, 1, 2)
    plt.imshow(data.transpose(), aspect='auto', extent=[t[0], t[-1], num_neurons, 0], interpolation='none')  # ,cmap=cmap,norm=norm)
    plt.xlabel('t (ms)')
    plt.ylabel('Neuron Index')

    return fig

def smooth_data(data, smooth_factor):
    data_smooth = np.zeros_like(data)
    num_rows = np.shape(data)[0]
    data_smooth[0,:] = data[0,:]
    for i in range(1,num_rows):
        data_smooth[i,:] = data[i,:]*smooth_factor + (1-smooth_factor)*data_smooth[i-1,:]
    return data_smooth

def run_pca(data, debug):
    pca = PCA()
    pca.fit(data)

    explained_variance_ratio = pca.explained_variance_ratio_

    if debug:
        print('First component explains %f %% of the variance' % (100*explained_variance_ratio[0]))
        print('Second component explains %f %% of the variance' % (100 * explained_variance_ratio[1]))
        print('Third component explains %f %% of the variance' % (100 * explained_variance_ratio[2]))

    component_0 = pca.components_[0, :]
    component_1 = pca.components_[1, :]
    component_2 = pca.components_[2, :]

    data_0 = np.matmul(data, component_0)
    data_1 = np.matmul(data, component_1)
    data_2 = np.matmul(data, component_2)

    return data_0, data_1, data_2, explained_variance_ratio

def plot_pca_space(t, data_0, data_1, data_2):
    fig = plt.figure()
    grid = fig.add_gridspec(2,3)

    ax = fig.add_subplot(grid[0,:])
    ax.plot(t, data_0, label='First Component')
    ax.plot(t, data_1, label='Second Component')
    ax.plot(t, data_2, label='Third Component')
    ax.legend()
    ax.set(title='Components over Time', xlabel='t (ms)', ylabel='Component Space')

    ax = fig.add_subplot(grid[1,0])
    ax.plot(data_0, data_1)
    ax.set(title='PCA: 2 Components', xlabel='First Component Space', ylabel='Second Component Space')

    ax = fig.add_subplot(grid[1,1:], projection='3d')
    ax.plot3D(data_0, data_1, data_2)
    ax.set(title='PCA: 3 Components', xlabel='First Component Space', ylabel='Second Component Space')

    return fig

def print_elapsed_time(start):
    print('%f seconds elapsed' % (time.time() - start))

def sort_data(data, reference, num_neurons):
    shifts = np.zeros(num_neurons)
    for i in range(num_neurons):
        xcorr = scipy.signal.correlate(reference, data[:,i])

        nsamples = len(reference)
        delta = np.arange(1 - nsamples, nsamples)
        shifts[i] = delta[xcorr.argmax()]

    shifts_reversed = shifts.argsort()[::-1]
    data_sorted = data[:, shifts_reversed]

    return data_sorted, shifts_reversed

def plot_sorted_data(data_sorted, data, t, num_neurons):
    fig = plt.figure()
    plt.subplot(2, 1, 1)
    plt.title('Raw Membrane Potentials')
    spike_raster_plot(t, data.transpose()[:][:])
    plt.subplot(2, 1, 2)
    plt.title('Sorted by Descending Phase Shift')
    spike_raster_plot(t, data_sorted.transpose()[:][:])
    return fig

def balanced_sequence_generator(g_max_ex=1.0, g_max_in=1.0, reversal_ex=2.5, reversal_in=-2.5, g_m=1.0, c_m=5.0,
                                rest=0.0, activity_range=1.0, num_neurons=200, ratio_conn=0.1, stim=3.0, dt=0.1, num_steps=10000,
                                seed=0, plot=True, backend='numpy', device='cpu', debug=True, smooth_factor=0.5, tau=1.0):
    start = time.time()
    # Initialize random generator
    if debug:
        print('Creating seeded random number generator at %f seconds')
        print_elapsed_time(start)
    rng = np.random.default_rng(seed=seed)

    # Create network
    if debug:
        print('\nCreating the SNS Network')
        print_elapsed_time(start)

    model = gen_network(num_neurons, ratio_conn, g_max_ex, g_max_in, reversal_ex, reversal_in, g_m, c_m, rest,
                            activity_range, dt, rng, backend, device, tau)

    # Run the network
    if debug:
        print('\nRunning the Network')
        print_elapsed_time(start)

    data, t = run_network(model, num_steps, stim, dt)

    data_smooth = smooth_data(data, smooth_factor)
    plt.figure()
    plt.imshow(data_smooth.transpose(), aspect='auto', extent=[t[0], t[-1], num_neurons, 0], interpolation='none')

    # Run Principal Component Analysis
    if debug:
        print('\nPerforming PCA')
        print_elapsed_time(start)

    data_0, data_1, data_2, explained_variance_ratio = run_pca(data_smooth, debug)

    # Sort the data by phase offset from the first principal component
    if debug:
        print('\nSorting the data by phase')
        print_elapsed_time(start)

    data_sorted, shifts = sort_data(data[-1000:,:], data_0[-1000:], num_neurons)

    if debug:
        print('\nFinished')
        print_elapsed_time(start)

    if plot:
        fig_connectivity = plot_connectivity(model)

        # Plot the full data
        fig_full_run = plot_activity(data, t, 'Full Simulation', num_neurons)

        # Plot the cumulative explained variance
        fig_explained_variance = plt.figure()
        plt.title('PCA Level of Explained Variance')
        plt.plot(np.cumsum(explained_variance_ratio))
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')

        # Plot the 2d and 3d principal component space over the whole trajectory
        fig_pca_full = plot_pca_space(t, data_0, data_1, data_2)

        # Plot the data for the last 1000 steps
        fig_last_1000 = plot_activity(data[-1000:, :], t[-1000:], 'Last 1000 Steps', num_neurons)

        # Plot the components over the last 1000 steps
        fig_pca_last_1000 = plot_pca_space(t[-1000:], data_0[-1000:], data_1[-1000:], data_2[-1000:])

        # Plot the sorted data
        fig_sorted_last_1000 = plot_sorted_data(data_sorted, data[-1000:,:], t[-1000:], num_neurons)


# Connection parameters
g_max_ex = 1.0
g_max_in = 1.0
reversal_ex = 2.5
reversal_in = -2.5
ratio_conn = 0.1
tau = 5.0

# Neuron parameters
g_m = 1.0
c_m = 5.0
rest = 0.0
activity_range = 1.0
num_neurons = 200

# Simulation settings
seed = 0
dt = 0.1
num_steps = 10000
backend = 'numpy'
device = 'cpu'

# Optional flags
plot = True
debug = True
smooth_factor = 0.01

stim = 2.5 # <------------ Change this to easily change behavior

balanced_sequence_generator(g_max_ex=g_max_ex, g_max_in=g_max_in, reversal_ex=reversal_ex, reversal_in=reversal_in,
                            g_m=g_m, c_m=c_m, rest=rest, activity_range=activity_range, num_neurons=num_neurons,
                            ratio_conn=ratio_conn, stim=stim, dt=dt, num_steps=num_steps, seed=seed, plot=plot,
                            backend=backend, device=device, debug=debug, smooth_factor=smooth_factor,tau=tau)

plt.show()