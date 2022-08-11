"""
Simulation backends for synthetic nervous system networks. Each of these are python-based, and are constructed using a
Network. They can then be run for a step, with the inputs being a vector of neural states and applied currents and the
output being the next step of neural states.
"""

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
IMPORTS
"""

from typing import Any
import numpy as np
import torch
import sys
import warnings

from sns_toolbox.networks import Network
from sns_toolbox.neurons import SpikingNeuron, NonSpikingNeuronWithGatedChannels

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
BASE CLASS
"""

class __Backend__:
    """
    Base-level class for all simulation backends. Each will construct a representation of a given network using the
    desired backend technique, and take in (some form of) a vector of input states and applied currents, and compute
    the result for the next timestep.

    :param network:     Network to serve as a design template.
    :type network:      sns_toolbox.design.networks.Network
    :param dt:          Simulation time constant, default is 0.1. Units are milliseconds (ms).
    :type dt:           Number, optional
    :param debug:       When enabled, print debug information to the console. Default is 'False'.
    :type debug:        bool, optional
    :param substeps:    Number of simulation substeps before returning an output vector. Default is 1.
    :type substeps:     int, optional
    """
    def __init__(self, network: Network, dt: float = 0.1, debug: bool = False, substeps: int = 1) -> None:

        if substeps <= 0:
            raise ValueError('Substeps must be a positive integer')
        self.substeps = substeps
        self.network = network
        self.dt = dt
        self.debug = debug
        self.spiking = network.params['spiking']
        self.delay = network.params['delay']
        self.electrical = network.params['electrical']
        self.electrical_rectified = network.params['electricalRectified']
        self.gated = network.params['gated']
        self.num_channels = network.params['numChannels']

        if self.debug:
            print('#\nGETTING NET PARAMETERS\n#')
        self.__get_net_params__()

        if self.debug:
            print('#\nINITIALIZING VARIABLES AND PARAMETERS\n#')
        self.__initialize_vectors_and_matrices__()

        if self.debug:
            print('#\nSETTING NEURAL PARAMETERS\n#')
        self.__set_neurons__()

        if self.debug:
            print('#\nSETTING INPUT PARAMETERS\n#')
        self.__set_inputs__()

        if self.debug:
            print('#\nSETTING CONNECTION PARAMETERS\n#')
        self.__set_connections__()

        if self.debug:
            print('#\nCALCULATING TIME FACTORS\n#')
        self.__calculate_time_factors__()

        if self.spiking:
            if self.delay:
                if self.debug:
                    print('#\nINITIALIZING PROPAGATION DELAY\n#')
                self.__initialize_propagation_delay__()

        if self.debug:
            print('#\nSETTING OUTPUT PARAMETERS\n#')
        self.__set_outputs__()

        if self.debug:
            print('#\nALL FINAL PARAMETERS\n#')
            self.__debug_print__()
            print('#\nDONE BUILDING\n#')

    def __get_net_params__(self) -> None:
        """
        Get the main properties from the network. These are the number of populations, number of neurons, number of
        connections, number of inputs, number of outputs, and the network range of neural activity.

        :return: None
        :rtype: N/A
        """
        self.num_populations = self.network.get_num_populations()
        self.num_neurons = self.network.get_num_neurons()
        self.num_connections = self.network.get_num_connections()
        self.num_inputs = self.network.get_num_inputs()
        self.num_outputs = self.network.get_num_outputs()
        self.R = self.network.params['R']

        if self.debug:
            print('Number of Populations:')
            print(self.num_populations)
            print('Number of Neurons:')
            print(self.num_neurons)
            print('Number of Connections')
            print(self.num_connections)
            print('Number of Inputs:')
            print(self.num_inputs)
            print('Number of Outputs:')
            print('Network Voltage Range (mV):')
            print(self.R)

    def __initialize_vectors_and_matrices__(self) -> None:
        """
        Initialize all the vectors and matrices needed for all the neural states and parameters. That includes the
        following: U, ULast, Spikes, Cm, Gm, Ibias, Theta0, Theta, ThetaLast, m, TauTheta.

        :return:    None
        :rtype:     N/A
        """
        raise NotImplementedError

    def __set_neurons__(self) -> None:
        """
        Iterate over all populations in the network, and set the corresponding neural parameters for each neuron in the
        network: Cm, Gm, Ibias, ULast, U, Theta0, ThetaLast, Theta, TauTheta, m.

        :return:    None
        :rtype:     N/A
        """
        raise NotImplementedError

    def __set_inputs__(self) -> None:
        """
        Build the input connection matrix.

        :return:    None
        :rtype:     N/A
        """
        raise NotImplementedError

    def __set_connections__(self) -> None:
        """
        Build the synaptic parameter matrices. Interpret connectivity patterns between populations into individual
        synapses.

        :return:    None
        :rtype:     N/A
        """
        raise NotImplementedError

    def __calculate_time_factors__(self) -> None:
        """
        Precompute the time factors for the membrane voltage, firing threshold, and spiking synapses.

        :return:    None
        :rtype:     N/A
        """
        self.time_factor_membrane = self.dt / (self.c_m / self.g_m)
        if self.spiking:
            self.time_factor_threshold = self.dt / self.tau_theta
            self.time_factor_synapse = self.dt / self.tau_syn

    def __initialize_propagation_delay__(self) -> None:
        """
        Create a buffer sized to store enough spike data for the longest synaptic propagation delay.

        :return:    None
        :rtype:     N/A
        """
        raise NotImplementedError

    def __set_outputs__(self) -> None:
        """
        Build the output connectivity matrices for voltage and spike monitors. Generate separate output monitors for
        each neuron in a population.

        :return:    None
        :rtype:     N/A
        """
        raise NotImplementedError

    def __debug_print__(self) -> None:
        """
        Print the values for every vector/matrix which will be used in the forward computation.

        :return:    None
        :rtype:     N/A
        """
        print('Input Connectivity:')
        print(self.input_connectivity)
        print('g_max_non:')
        print(self.g_max_non)
        if self.spiking:
            print('GmaxSpike:')
            print(self.g_max_spike)
        print('del_e:')
        print(self.del_e)
        if self.electrical:
            print('Gelectrical:')
            print(self.g_electrical)
        if self.electrical_rectified:
            print('GelectricalRectified:')
            print(self.g_rectified)
        print('Output Voltage Connectivity')
        print(self.output_voltage_connectivity)
        if self.spiking:
            print('Output Spike Connectivity:')
            print(self.output_spike_connectivity)
        print('u:')
        print(self.u)
        print('u_last:')
        print(self.u_last)
        if self.spiking:
            print('theta_0:')
            print(self.theta_0)
            print('ThetaLast:')
            print(self.theta_last)
            print('Theta')
            print(self.theta)
        if self.gated:
            print('Number of Channels:')
            print(self.num_channels)
            print('Ionic Conductance:')
            print(self.g_ion)
            print('Ionic Reversal Potentials:')
            print(self.e_ion)
            print('A Gate Parameters:')
            print('Power:')
            print(self.pow_a)
            print('Slope:')
            print(self.slope_a)
            print('K:')
            print(self.k_a)
            print('Reversal Potential:')
            print(self.e_a)
            print('B Gate Parameters:')
            print('Power:')
            print(self.pow_b)
            print('Slope:')
            print(self.slope_b)
            print('K:')
            print(self.k_b)
            print('Reversal Potential:')
            print(self.e_b)
            print('Tau Max:')
            print(self.tau_max_b)
            print('B:')
            print(self.b_gate)
            print('B_last:')
            print(self.b_gate_last)
            print('C Gate Parameters:')
            print('Power:')
            print(self.pow_c)
            print('Slope:')
            print(self.slope_c)
            print('K:')
            print(self.k_c)
            print('Reversal Potential:')
            print(self.e_c)
            print('Tau Max:')
            print(self.tau_max_c)
            print('B:')
            print(self.c_gate)
            print('B_last:')
            print(self.c_gate_last)

    def reset(self, u=None, theta=None, b_gate=None, c_gate=None) -> None:
        """
        Reset the network, either to its initial state or an arbitrary state.

        :param u:       State for u. Default is None.
        :param theta:   State for theta. Default is None.
        :param b_gate:  State for the b gate. Default is None.
        :param c_gate:  State for the c gate. Default is None.
        :return: None
        """
        raise NotImplementedError

    def forward(self, inputs) -> Any:
        """
        Compute the next neural states based on previous neural states. Handle substeps as well.

        :param inputs:  Input currents into the network.
        :type inputs:   np.ndarray or torch.tensor
        :return:        The neural states at the next step.
        :rtype:         np.ndarray or torch.tensor
        """
        for i in range(self.substeps):
            out = self.__forward_pass__(inputs)
        return out

    def __forward_pass__(self, inputs) -> Any:
        """
        Compute the next neural states based on previous neural states in the following steps:
        Ulast = U;
        ThetaLast = Theta;
        MappedInputs = cubic*inputs^3 + quadratic*inputs^2 + linear*inputs + offset;
        IApp = InputConnectivity X MappedInputs;
        GNon = max(0, min(GMaxNon*ULast/R, GMaxNon));
        GSpike = GSpike * (1-TimeFactorSynapse);
        GSyn = GNon + GSpike;
        ISyn = ColSum(GSyn*DelE) - ULast*ColSum(GSyn);
        U = ULast + TimeFactorMembrane*(-Gm*ULast + IBias + ISyn + IApp);
        Theta = ThetaLast + TimeFactorThreshold*(-ThetaLast + Theta0 + m*ULast);
        Spikes = Sign(min(0, Theta - U));
        SpikeBuffer = SpikeBuffer shifted down by 1;
        SpikeBuffer[first row] = Spikes;
        DelayedSpikeMatrix = SpikeBuffer[BufferSteps, BufferedNeurons];
        GSpike = max(GSpike, -DelayedSpikeMatrix*GMaxSpike);
        U = U * (Spikes + 1);
        Outputs = OutputVoltageConnectivity X U + OutputSpikeConnectivity X (-Spikes);
        MappedOutputs = cubic*Outputs^3 + quadratic*Outputs^2 + linear*Outputs + offset.

        :param inputs:  Input currents into the network.
        :type inputs:   np.ndarray or torch.tensor
        :return:        The neural states at the next step.
        :rtype:         np.ndarray or torch.tensor
        """
        raise NotImplementedError

"""
########################################################################################################################
NUMPY BACKEND

Simulating the network using numpy vectors and matrices.
Note that this is not sparse, so memory may explode for large networks
"""

class SNS_Numpy(__Backend__):
    """
    Simulation backend based in Numpy.

    :param network: Network which will be compiled to Numpy.
    :type network:  sns_toolbox.design.networks.Network
    """
    def __init__(self,network: Network,**kwargs):
        super().__init__(network,**kwargs)

    def __initialize_vectors_and_matrices__(self) -> None:

        self.u = np.zeros(self.num_neurons)
        self.u_last = np.zeros(self.num_neurons)
        self.u_0 = np.zeros(self.num_neurons)
        self.c_m = np.zeros(self.num_neurons)
        self.g_m = np.zeros(self.num_neurons)
        self.i_b = np.zeros(self.num_neurons)
        if self.spiking:
            self.spikes = np.zeros(self.num_neurons)
            self.theta_0 = np.zeros(self.num_neurons)
            self.theta = np.zeros(self.num_neurons)
            self.theta_last = np.zeros(self.num_neurons)
            self.m = np.zeros(self.num_neurons)
            self.tau_theta = np.zeros(self.num_neurons)

        self.g_max_non = np.zeros([self.num_neurons, self.num_neurons])
        self.del_e = np.zeros([self.num_neurons, self.num_neurons])
        if self.spiking:
            self.g_max_spike = np.zeros([self.num_neurons, self.num_neurons])
            self.g_spike = np.zeros([self.num_neurons, self.num_neurons])
            self.tau_syn = np.zeros([self.num_neurons, self.num_neurons]) + 1
            if self.delay:
                self.spike_delays = np.zeros([self.num_neurons, self.num_neurons])
                self.spike_rows = []
                self.spike_cols = []
                self.buffer_steps = []
                self.buffer_nrns = []
                self.delayed_spikes = np.zeros([self.num_neurons, self.num_neurons])
        if self.electrical:
            self.g_electrical = np.zeros([self.num_neurons, self.num_neurons])
        if self.electrical_rectified:
            self.g_rectified = np.zeros([self.num_neurons, self.num_neurons])
        if self.gated:
            # Channel params
            self.g_ion = np.zeros([self.num_channels, self.num_neurons])
            self.e_ion = np.zeros([self.num_channels, self.num_neurons])
            # A gate params
            self.pow_a = np.zeros([self.num_channels, self.num_neurons])
            self.slope_a = np.zeros([self.num_channels, self.num_neurons])
            self.k_a = np.zeros([self.num_channels, self.num_neurons])+1
            self.e_a = np.zeros([self.num_channels, self.num_neurons])
            # B gate params
            self.pow_b = np.zeros([self.num_channels, self.num_neurons])
            self.slope_b = np.zeros([self.num_channels, self.num_neurons])
            self.k_b = np.zeros([self.num_channels, self.num_neurons])+1
            self.e_b = np.zeros([self.num_channels, self.num_neurons])
            self.tau_max_b = np.zeros([self.num_channels, self.num_neurons])+1
            # C gate params
            self.pow_c = np.zeros([self.num_channels, self.num_neurons])
            self.slope_c = np.zeros([self.num_channels, self.num_neurons])
            self.k_c = np.zeros([self.num_channels, self.num_neurons])+1
            self.e_c = np.zeros([self.num_channels, self.num_neurons])
            self.tau_max_c = np.zeros([self.num_channels, self.num_neurons])+1

            self.b_gate = np.zeros([self.num_channels, self.num_neurons])
            self.b_gate_last = np.zeros([self.num_channels, self.num_neurons])
            self.b_gate_0 = np.zeros([self.num_channels, self.num_neurons])
            self.c_gate = np.zeros([self.num_channels, self.num_neurons])
            self.c_gate_last = np.zeros([self.num_channels, self.num_neurons])
            self.c_gate_0 = np.zeros([self.num_channels, self.num_neurons])

        self.pops_and_nrns = []
        index = 0
        for pop in range(len(self.network.populations)):
            num_neurons = self.network.populations[pop]['number']  # find the number of neurons in the population
            self.pops_and_nrns.append([])
            for num in range(num_neurons):
                self.pops_and_nrns[pop].append(index)
                index += 1

    def __set_neurons__(self) -> None:

        index = 0
        for pop in range(len(self.network.populations)):
            num_neurons = self.network.populations[pop]['number']  # find the number of neurons in the population
            initial_value = self.network.populations[pop]['initial_value']
            for num in range(num_neurons):  # for each neuron, copy the parameters over
                self.c_m[index] = self.network.populations[pop]['type'].params['membrane_capacitance']
                self.g_m[index] = self.network.populations[pop]['type'].params['membrane_conductance']
                self.i_b[index] = self.network.populations[pop]['type'].params['bias']
                if hasattr(initial_value, '__iter__'):
                    self.u_last[index] = initial_value[num]
                elif initial_value is None:
                    self.u_last[index] = 0.0
                else:
                    self.u_last[index] = initial_value
                if self.spiking:
                    if isinstance(self.network.populations[pop]['type'], SpikingNeuron):  # if the neuron is spiking, copy more
                        self.theta_0[index] = self.network.populations[pop]['type'].params['threshold_initial_value']
                        self.m[index] = self.network.populations[pop]['type'].params['threshold_proportionality_constant']
                        self.tau_theta[index] = self.network.populations[pop]['type'].params['threshold_time_constant']
                    else:  # otherwise, set to the special values for NonSpiking
                        self.theta_0[index] = sys.float_info.max
                        self.m[index] = 0
                        self.tau_theta[index] = 1
                if self.gated:
                    if isinstance(self.network.populations[pop]['type'], NonSpikingNeuronWithGatedChannels):
                        # Channel params
                        self.g_ion[:, index] = self.network.populations[pop]['type'].params['Gion']
                        self.e_ion[:, index] = self.network.populations[pop]['type'].params['Eion']
                        # A gate params
                        self.pow_a[:, index] = self.network.populations[pop]['type'].params['paramsA']['pow']
                        self.slope_a[:, index] = self.network.populations[pop]['type'].params['paramsA']['slope']
                        self.k_a[:, index] = self.network.populations[pop]['type'].params['paramsA']['k']
                        self.e_a[:, index] = self.network.populations[pop]['type'].params['paramsA']['reversal']
                        # B gate params
                        self.pow_b[:, index] = self.network.populations[pop]['type'].params['paramsB']['pow']
                        self.slope_b[:, index] = self.network.populations[pop]['type'].params['paramsB']['slope']
                        self.k_b[:, index] = self.network.populations[pop]['type'].params['paramsB']['k']
                        self.e_b[:, index] = self.network.populations[pop]['type'].params['paramsB']['reversal']
                        self.tau_max_b[:, index] = self.network.populations[pop]['type'].params['paramsB']['TauMax']
                        # C gate params
                        self.pow_c[:, index] = self.network.populations[pop]['type'].params['paramsC']['pow']
                        self.slope_c[:, index] = self.network.populations[pop]['type'].params['paramsC']['slope']
                        self.k_c[:, index] = self.network.populations[pop]['type'].params['paramsC']['k']
                        self.e_c[:, index] = self.network.populations[pop]['type'].params['paramsC']['reversal']
                        self.tau_max_c[:, index] = self.network.populations[pop]['type'].params['paramsC']['TauMax']

                        self.b_gate_last[:, index] = 1 / (1 + self.k_b[:, index] * np.exp(self.slope_b[:, index]*(self.u_last[index]-self.e_b[:, index])))
                        self.c_gate_last[:, index] = 1 / (1 + self.k_c[:, index] * np.exp(self.slope_c[:, index]*(self.u_last[index]-self.e_c[:, index])))
                index += 1
        self.u = np.copy(self.u_last)
        self.u_0 = np.copy(self.u_last)
        if self.spiking:
            self.theta = np.copy(self.theta_0)
            self.theta_last = np.copy(self.theta_0)
        if self.gated:
            self.b_gate = np.copy(self.b_gate_last)
            self.b_gate_0 = np.copy(self.b_gate_last)
            self.c_gate = np.copy(self.c_gate_last)
            self.c_gate_0 = np.copy(self.c_gate_last)

    def __set_inputs__(self) -> None:

        self.input_connectivity = np.zeros([self.num_neurons, self.network.get_num_inputs_actual()])  # initialize connectivity matrix
        index = 0
        for inp in range(self.network.get_num_inputs()):  # iterate over the connections in the network
            size = self.network.inputs[inp]['size']
            dest_pop = self.network.inputs[inp]['destination']  # get the destination
            if size == 1:
                for dest in self.pops_and_nrns[dest_pop]:
                    self.input_connectivity[dest][inp] = 1.0  # set the weight in the correct source and destination
                index += 1
            else:
                for dest in self.pops_and_nrns[dest_pop]:
                    self.input_connectivity[dest][index] = 1.0
                    index += 1

    def __set_connections__(self) -> None:

        for syn in range(len(self.network.connections)):
            source_pop = self.network.connections[syn]['source']
            dest_pop = self.network.connections[syn]['destination']
            g_max = self.network.connections[syn]['params']['max_conductance']
            if self.network.connections[syn]['params']['electrical'] is False: # electrical connection
                del_e = self.network.connections[syn]['params']['reversal_potential']

            if self.network.connections[syn]['params']['pattern']:  # pattern connection
                pop_size = len(self.pops_and_nrns[source_pop])
                source_index = self.pops_and_nrns[source_pop][0]
                dest_index = self.pops_and_nrns[dest_pop][0]
                if self.network.connections[syn]['params']['spiking']:
                    tau_s = self.network.connections[syn]['params']['synapticTimeConstant']
                    self.g_max_spike[dest_index:dest_index+pop_size,source_index:source_index+pop_size] = g_max
                    self.del_e[dest_index:dest_index+pop_size,source_index:source_index+pop_size] = del_e
                    self.tau_syn[dest_index:dest_index+pop_size,source_index:source_index+pop_size] = tau_s
                    if self.delay:
                        delay = self.network.connections[syn]['params']['synapticTransmissionDelay']
                        self.spike_delays[dest_index:dest_index+pop_size,source_index:source_index+pop_size] = delay

                        for source in self.pops_and_nrns[source_pop]:
                            for dest in self.pops_and_nrns[dest_pop]:
                                self.buffer_nrns.append(source)
                                self.buffer_steps.append(delay)
                                self.spike_rows.append(dest)
                                self.spike_cols.append(source)
                else:
                    self.g_max_non[dest_index:dest_index+pop_size,source_index:source_index+pop_size] = g_max
                    self.del_e[dest_index:dest_index+pop_size,source_index:source_index+pop_size] = del_e
            elif self.network.connections[syn]['params']['electrical']: # electrical connection
                for source in self.pops_and_nrns[source_pop]:
                    for dest in self.pops_and_nrns[dest_pop]:
                        if self.network.connections[syn]['params']['rectified']:    # rectified
                            self.g_rectified[dest][source] = g_max / len(self.pops_and_nrns[source_pop])
                        else:
                            self.g_electrical[dest][source] = g_max / len(self.pops_and_nrns[source_pop])
                            self.g_electrical[source][dest] = g_max / len(self.pops_and_nrns[source_pop])
            else:   # chemical connection
                if self.network.connections[syn]['params']['spiking']:  # spiking chemical synapse
                    tau_s = self.network.connections[syn]['params']['synapticTimeConstant']
                    if self.delay:
                        delay = self.network.connections[syn]['params']['synapticTransmissionDelay']
                    for source in self.pops_and_nrns[source_pop]:
                        for dest in self.pops_and_nrns[dest_pop]:
                            self.g_max_spike[dest][source] = g_max / len(self.pops_and_nrns[source_pop])
                            self.del_e[dest][source] = del_e
                            self.tau_syn[dest][source] = tau_s
                            if self.delay:
                                self.spike_delays[dest][source] = delay
                                self.buffer_nrns.append(source)
                                self.buffer_steps.append(delay)
                                self.spike_rows.append(dest)
                                self.spike_cols.append(source)
                else:   # nonspiking chemical synapse
                    for source in self.pops_and_nrns[source_pop]:
                        for dest in self.pops_and_nrns[dest_pop]:
                            self.g_max_non[dest][source] = g_max / len(self.pops_and_nrns[source_pop])
                            self.del_e[dest][source] = del_e

    def __initialize_propagation_delay__(self) -> None:
        buffer_length = int(np.max(self.spike_delays) + 1)
        self.spike_buffer = np.zeros([buffer_length, self.num_neurons])

    def __set_outputs__(self) -> None:

        outputs = []
        index = 0
        for out in range(len(self.network.outputs)):
            source_pop = self.network.outputs[out]['source']
            num_source_neurons = self.network.populations[source_pop]['number']
            outputs.append([])
            for num in range(num_source_neurons):
                outputs[out].append(index)
                index += 1
        self.num_outputs = index

        self.output_voltage_connectivity = np.zeros(
            [self.num_outputs, self.num_neurons])  # initialize connectivity matrix
        if self.spiking:
            self.output_spike_connectivity = np.copy(self.output_voltage_connectivity)
        self.outputs = np.zeros(self.num_outputs)
        for out in range(len(self.network.outputs)):  # iterate over the connections in the network
            source_pop = self.network.outputs[out]['source']  # get the source
            for i in range(len(self.pops_and_nrns[source_pop])):
                if self.network.outputs[out]['spiking']:
                    self.output_spike_connectivity[outputs[out][i]][
                        self.pops_and_nrns[source_pop][i]] = 1.0  # set the weight in the correct source and destination
                else:
                    self.output_voltage_connectivity[outputs[out][i]][
                        self.pops_and_nrns[source_pop][i]] = 1.0  # set the weight in the correct source and destination

    def reset(self, u=None, theta=None, b_gate=None, c_gate=None) -> None:
        if u is None:
            self.u = np.copy(self.u_0)
            self.u_last = np.copy(self.u_0)
        else:
            self.u = np.copy(u)
            self.u_last = np.copy(u)
        if self.spiking:
            if theta is None:
                self.theta = np.copy(self.theta_0)
                self.theta_last = np.copy(self.theta_0)
            else:
                self.theta = np.copy(theta)
                self.theta_last = np.copy(theta)
        if self.gated:
            if b_gate is None:
                self.b_gate = np.copy(self.b_gate_0)
                self.b_gate_last = np.copy(self.b_gate_0)
            else:
                self.b_gate = np.copy(b_gate)
                self.b_gate_last = np.copy(b_gate)
            if c_gate is None:
                self.c_gate = np.copy(self.c_gate_0)
                self.c_gate_last = np.copy(self.c_gate_0)
            else:
                self.c_gate = np.copy(c_gate)
                self.c_gate_last = np.copy(c_gate)

    def __forward_pass__(self, inputs) -> Any:
        self.u_last = np.copy(self.u)
        i_app = np.matmul(self.input_connectivity, inputs)  # Apply external current sources to their destinations
        g_syn = np.maximum(0, np.minimum(self.g_max_non * self.u_last / self.R, self.g_max_non))
        if self.spiking:
            self.theta_last = np.copy(self.theta)
            self.g_spike = self.g_spike * (1 - self.time_factor_synapse)
            g_syn += self.g_spike

        i_syn = np.sum(g_syn * self.del_e, axis=1) - self.u_last * np.sum(g_syn, axis=1)
        if self.electrical:
            i_syn += (np.sum(self.g_electrical*self.u_last, axis=1) - self.u_last*np.sum(self.g_electrical, axis=1))
        if self.electrical_rectified:
            # create mask
            mask = np.subtract.outer(self.u_last, self.u_last).transpose() > 0
            masked_g = mask*self.g_rectified
            diag_masked = masked_g + masked_g.transpose() - np.diag(masked_g.diagonal())
            i_syn += np.sum(diag_masked*self.u_last, axis=1) - self.u_last*np.sum(diag_masked, axis=1)
        if self.gated:
            a_inf = 1 / (1 + self.k_a * np.exp(self.slope_a*(self.e_a-self.u_last)))
            b_inf = 1 / (1 + self.k_b * np.exp(self.slope_b*(self.e_b-self.u_last)))
            c_inf = 1 / (1 + self.k_c * np.exp(self.slope_c*(self.e_c-self.u_last)))

            tau_b = self.tau_max_b * b_inf * np.sqrt(self.k_b*np.exp(self.slope_b*(self.e_b-self.u_last)))
            tau_c = self.tau_max_c * c_inf * np.sqrt(self.k_c*np.exp(self.slope_c*(self.e_c-self.u_last)))

            self.b_gate_last = np.copy(self.b_gate)
            self.c_gate_last = np.copy(self.c_gate)

            self.b_gate = self.b_gate_last + self.dt * ((b_inf - self.b_gate_last) / tau_b)
            self.c_gate = self.c_gate_last + self.dt * ((c_inf - self.c_gate_last) / tau_c)

            i_ion = self.g_ion*(a_inf**self.pow_a)*(self.b_gate**self.pow_b)*(self.c_gate**self.pow_c)*(self.e_ion-self.u_last)
            i_gated = np.sum(i_ion, axis=0)

            self.u = self.u_last + self.time_factor_membrane * (-self.g_m * self.u_last + self.i_b + i_syn + i_app + i_gated)  # Update membrane potential
        else:
            self.u = self.u_last + self.time_factor_membrane * (-self.g_m * self.u_last + self.i_b + i_syn + i_app)  # Update membrane potential
        if self.spiking:
            self.theta = self.theta_last + self.time_factor_threshold * (-self.theta_last + self.theta_0 + self.m * self.u_last)  # Update the firing thresholds
            self.spikes = np.sign(np.minimum(0, self.theta - self.u))  # Compute which neurons have spiked

            # New stuff with delay
            if self.delay:
                self.spike_buffer = np.roll(self.spike_buffer, 1, axis=0)   # Shift buffer entries down
                self.spike_buffer[0, :] = self.spikes    # Replace row 0 with the current spike data
                # Update a matrix with all of the appropriately delayed spike values
                self.delayed_spikes[self.spike_rows, self.spike_cols] = self.spike_buffer[self.buffer_steps, self.buffer_nrns]

                self.g_spike = np.maximum(self.g_spike, (-self.delayed_spikes) * self.g_max_spike)  # Update the conductance of connections which spiked
            else:
                self.g_spike = np.maximum(self.g_spike, (-self.spikes) * self.g_max_spike)  # Update the conductance of connections which spiked
            self.u = self.u * (self.spikes + 1)  # Reset the membrane voltages of neurons which spiked
        self.outputs = np.matmul(self.output_voltage_connectivity, self.u)
        if self.spiking:
            self.outputs += np.matmul(self.output_spike_connectivity, -self.spikes)

        return self.outputs



"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
PYTORCH DENSE

Simulating the network using GPU-compatible tensors.
Note that this is not sparse, so memory may explode for large networks
"""


class SNS_Torch(__Backend__):
    """
    Simulation backend based in PyTorch. In future versions different options will be toggled automatically, but for now
    are implemented as boolean flags.

    :param network: Network which will be compiled to PyTorch.
    :type network:  sns_toolbox.design.networks.Network
    :param device:  Device network will be stored on, default is 'cuda' (GPU).
    :type device:   str, optional
    """
    def __init__(self,network: Network,device: str = 'cuda',**kwargs):
        if device != 'cpu':
            if not torch.cuda.is_available():
                warnings.warn('CUDA Device Unavailable. Using CPU Instead')
                device = 'cpu'
        self.device = device
        super().__init__(network,**kwargs)

    def __initialize_vectors_and_matrices__(self) -> None:
        """
        Initialize all of the vectors and matrices needed for all of the neural states and parameters. That includes the
        following: U, ULast, Spikes, Cm, Gm, Ibias, Theta0, Theta, ThetaLast, m, TauTheta.
        :return:    None
        """
        self.u = torch.zeros(self.num_neurons,device=self.device)
        self.u_last = torch.zeros(self.num_neurons,device=self.device)
        self.u_0 = torch.zeros(self.num_neurons,device=self.device)
        self.c_m = torch.zeros(self.num_neurons,device=self.device)
        self.g_m = torch.zeros(self.num_neurons,device=self.device)
        self.i_b = torch.zeros(self.num_neurons,device=self.device)
        if self.spiking:
            self.spikes = torch.zeros(self.num_neurons, device=self.device)
            self.theta_0 = torch.zeros(self.num_neurons,device=self.device)
            self.theta = torch.zeros(self.num_neurons,device=self.device)
            self.theta_last = torch.zeros(self.num_neurons,device=self.device)
            self.m = torch.zeros(self.num_neurons,device=self.device)
            self.tau_theta = torch.zeros(self.num_neurons,device=self.device)

        self.g_max_non = torch.zeros([self.num_neurons, self.num_neurons],device=self.device)
        self.del_e = torch.zeros([self.num_neurons, self.num_neurons],device=self.device)
        if self.spiking:
            self.g_max_spike = torch.zeros([self.num_neurons, self.num_neurons], device=self.device)
            self.g_spike = torch.zeros([self.num_neurons, self.num_neurons], device=self.device)
            self.tau_syn = torch.ones([self.num_neurons, self.num_neurons],device=self.device)
            if self.delay:
                self.spike_delays = torch.zeros([self.num_neurons, self.num_neurons],device=self.device)
                self.spike_rows = []
                self.spike_cols = []
                self.buffer_steps = []
                self.buffer_nrns = []
                self.delayed_spikes = torch.zeros([self.num_neurons, self.num_neurons],device=self.device)
        if self.electrical:
            self.g_electrical = torch.zeros([self.num_neurons, self.num_neurons],device=self.device)
        if self.electrical_rectified:
            self.g_rectified = torch.zeros([self.num_neurons, self.num_neurons],device=self.device)
        if self.gated:
            # Channel params
            self.g_ion = torch.zeros([self.num_channels, self.num_neurons],device=self.device)
            self.e_ion = torch.zeros([self.num_channels, self.num_neurons],device=self.device)
            # A gate params
            self.pow_a = torch.zeros([self.num_channels, self.num_neurons],device=self.device)
            self.slope_a = torch.zeros([self.num_channels, self.num_neurons],device=self.device)
            self.k_a = torch.ones([self.num_channels, self.num_neurons],device=self.device)
            self.e_a = torch.zeros([self.num_channels, self.num_neurons],device=self.device)
            # B gate params
            self.pow_b = torch.zeros([self.num_channels, self.num_neurons],device=self.device)
            self.slope_b = torch.zeros([self.num_channels, self.num_neurons],device=self.device)
            self.k_b = torch.ones([self.num_channels, self.num_neurons],device=self.device)
            self.e_b = torch.zeros([self.num_channels, self.num_neurons],device=self.device)
            self.tau_max_b = torch.ones([self.num_channels, self.num_neurons],device=self.device)
            # C gate params
            self.pow_c = torch.zeros([self.num_channels, self.num_neurons],device=self.device)
            self.slope_c = torch.zeros([self.num_channels, self.num_neurons],device=self.device)
            self.k_c = torch.ones([self.num_channels, self.num_neurons],device=self.device)
            self.e_c = torch.zeros([self.num_channels, self.num_neurons],device=self.device)
            self.tau_max_c = torch.ones([self.num_channels, self.num_neurons],device=self.device)+1

            self.b_gate = torch.zeros([self.num_channels, self.num_neurons],device=self.device)
            self.b_gate_last = torch.zeros([self.num_channels, self.num_neurons],device=self.device)
            self.b_gate_0 = torch.zeros([self.num_channels, self.num_neurons], device=self.device)
            self.c_gate = torch.zeros([self.num_channels, self.num_neurons],device=self.device)
            self.c_gate_last = torch.zeros([self.num_channels, self.num_neurons],device=self.device)
            self.c_gate_0 = torch.zeros([self.num_channels, self.num_neurons], device=self.device)

        self.pops_and_nrns = []
        index = 0
        for pop in range(len(self.network.populations)):
            num_neurons = self.network.populations[pop]['number']  # find the number of neurons in the population
            self.pops_and_nrns.append([])
            for num in range(num_neurons):
                self.pops_and_nrns[pop].append(index)
                index += 1

    def __set_neurons__(self) -> None:
        """
        Iterate over all populations in the network, and set the corresponding neural parameters for each neuron in the
        network: Cm, Gm, Ibias, ULast, U, Theta0, ThetaLast, Theta, TauTheta, m.
        :return:
        """
        index = 0
        for pop in range(len(self.network.populations)):
            num_neurons = self.network.populations[pop]['number']  # find the number of neurons in the population
            initial_value = self.network.populations[pop]['initial_value']
            for num in range(num_neurons):  # for each neuron, copy the parameters over
                self.c_m[index] = self.network.populations[pop]['type'].params['membrane_capacitance']
                self.g_m[index] = self.network.populations[pop]['type'].params['membrane_conductance']
                self.i_b[index] = self.network.populations[pop]['type'].params['bias']
                if hasattr(initial_value, '__iter__'):
                    self.u_last[index] = initial_value[num]
                elif initial_value is None:
                    self.u_last[index] = 0.0
                else:
                    self.u_last[index] = initial_value

                if self.spiking:
                    if isinstance(self.network.populations[pop]['type'], SpikingNeuron):  # if the neuron is spiking, copy more
                        self.theta_0[index] = self.network.populations[pop]['type'].params['threshold_initial_value']
                        self.m[index] = self.network.populations[pop]['type'].params['threshold_proportionality_constant']
                        self.tau_theta[index] = self.network.populations[pop]['type'].params['threshold_time_constant']
                    else:  # otherwise, set to the special values for NonSpiking
                        self.theta_0[index] = torch.finfo(self.theta_0[index].dtype).max
                        self.m[index] = 0
                        self.tau_theta[index] = 1
                if self.gated:
                    if isinstance(self.network.populations[pop]['type'], NonSpikingNeuronWithGatedChannels):
                        # Channel params
                        self.g_ion[:, index] = self.network.populations[pop]['type'].params['Gion']
                        self.e_ion[:, index] = self.network.populations[pop]['type'].params['Eion']
                        # A gate params
                        self.pow_a[:, index] = self.network.populations[pop]['type'].params['paramsA']['pow']
                        self.slope_a[:, index] = self.network.populations[pop]['type'].params['paramsA']['slope']
                        self.k_a[:, index] = self.network.populations[pop]['type'].params['paramsA']['k']
                        self.e_a[:, index] = self.network.populations[pop]['type'].params['paramsA']['reversal']
                        # B gate params
                        self.pow_b[:, index] = self.network.populations[pop]['type'].params['paramsB']['pow']
                        self.slope_b[:, index] = self.network.populations[pop]['type'].params['paramsB']['slope']
                        self.k_b[:, index] = self.network.populations[pop]['type'].params['paramsB']['k']
                        self.e_b[:, index] = self.network.populations[pop]['type'].params['paramsB']['reversal']
                        self.tau_max_b[:, index] = self.network.populations[pop]['type'].params['paramsB']['TauMax']
                        # C gate params
                        self.pow_c[:, index] = self.network.populations[pop]['type'].params['paramsC']['pow']
                        self.slope_c[:, index] = self.network.populations[pop]['type'].params['paramsC']['slope']
                        self.k_c[:, index] = self.network.populations[pop]['type'].params['paramsC']['k']
                        self.e_c[:, index] = self.network.populations[pop]['type'].params['paramsC']['reversal']
                        self.tau_max_c[:, index] = self.network.populations[pop]['type'].params['paramsC']['TauMax']

                        self.b_gate_last[:, index] = 1 / (1 + self.k_b[:, index] * np.exp(self.slope_b[:, index]*(self.u_last[index]-self.e_b[:, index])))
                        self.c_gate_last[:, index] = 1 / (1 + self.k_c[:, index] * np.exp(self.slope_c[:, index]*(self.u_last[index]-self.e_c[:, index])))
                index += 1
        self.u = self.u_last.clone()
        if self.spiking:
            self.theta = self.theta_0.clone()
            self.theta_last = self.theta_0.clone()
        if self.gated:
            self.b_gate = torch.clone(self.b_gate_last)
            self.c_gate = torch.clone(self.c_gate_last)
            self.b_gate_0 = torch.clone(self.b_gate_last)
            self.c_gate_0 = torch.clone(self.c_gate_last)

    def __set_inputs__(self) -> None:
        """
        Build the input connection matrix, and apply linear mapping coefficients.
        :return:    None
        """
        self.input_connectivity = torch.zeros([self.num_neurons, self.network.get_num_inputs_actual()],device=self.device)  # initialize connectivity matrix
        index = 0
        for inp in range(self.network.get_num_inputs()):  # iterate over the connections in the network
            size = self.network.inputs[inp]['size']
            dest_pop = self.network.inputs[inp]['destination']  # get the destination
            if size == 1:
                for dest in self.pops_and_nrns[dest_pop]:
                    self.input_connectivity[dest][inp] = 1.0  # set the weight in the correct source and destination
                index += 1
            else:
                for dest in self.pops_and_nrns[dest_pop]:
                    self.input_connectivity[dest][index] = 1.0
                    index += 1

    def __set_connections__(self) -> None:
        """
        Build the synaptic parameter matrices. Interpret connectivity patterns between populations into individual
        synapses.
        :return: None
        """
        for syn in range(len(self.network.connections)):
            source_pop = self.network.connections[syn]['source']
            dest_pop = self.network.connections[syn]['destination']
            g_max = self.network.connections[syn]['params']['max_conductance']
            if self.network.connections[syn]['params']['electrical'] is False:  # Chemical connection
                del_e = self.network.connections[syn]['params']['reversal_potential']

            if self.network.connections[syn]['params']['pattern']:  # pattern connection
                pop_size = len(self.pops_and_nrns[source_pop])
                source_index = self.pops_and_nrns[source_pop][0]
                dest_index = self.pops_and_nrns[dest_pop][0]

                if self.network.connections[syn]['params']['spiking']:
                    tau_s = self.network.connections[syn]['params']['synapticTimeConstant']
                    self.g_max_spike[dest_index:dest_index + pop_size,source_index:source_index + pop_size] = torch.from_numpy(g_max)
                    self.del_e[dest_index:dest_index + pop_size,source_index:source_index + pop_size] = torch.from_numpy(del_e)
                    self.tau_syn[dest_index:dest_index+pop_size,source_index:source_index+pop_size] = torch.from_numpy(tau_s)
                    if self.delay:
                        delay = self.network.connections[syn]['params']['synapticTransmissionDelay']
                        self.spike_delays[dest_index:dest_index+pop_size,source_index:source_index+pop_size] = torch.from_numpy(delay)

                        for source in self.pops_and_nrns[source_pop]:
                            for dest in self.pops_and_nrns[dest_pop]:
                                self.buffer_nrns.append(source)
                                self.buffer_steps.append(delay)
                                self.spike_rows.append(dest)
                                self.spike_cols.append(source)
                else:
                    self.g_max_non[dest_index:dest_index+pop_size,source_index:source_index+pop_size] = torch.from_numpy(g_max)
                    self.del_e[dest_index:dest_index+pop_size,source_index:source_index+pop_size] = torch.from_numpy(del_e)
            elif self.network.connections[syn]['params']['electrical']:  # electrical connection
                for source in self.pops_and_nrns[source_pop]:
                    for dest in self.pops_and_nrns[dest_pop]:
                        if self.network.connections[syn]['params']['rectified']:  # rectified
                            self.g_rectified[dest][source] = g_max / len(self.pops_and_nrns[source_pop])
                        else:
                            self.g_electrical[dest][source] = g_max / len(self.pops_and_nrns[source_pop])
                            self.g_electrical[source][dest] = g_max / len(self.pops_and_nrns[source_pop])
            else:   # chemical connection
                if self.network.connections[syn]['params']['spiking']:  # spiking chemical synapse
                    tau_s = self.network.connections[syn]['params']['synapticTimeConstant']
                    if self.delay:
                        delay = self.network.connections[syn]['params']['synapticTransmissionDelay']
                    for source in self.pops_and_nrns[source_pop]:
                        for dest in self.pops_and_nrns[dest_pop]:
                            self.g_max_spike[dest][source] = g_max / len(self.pops_and_nrns[source_pop])
                            self.del_e[dest][source] = del_e
                            self.tau_syn[dest][source] = tau_s
                            if self.delay:
                                self.spike_delays[dest][source] = delay
                                self.buffer_nrns.append(source)
                                self.buffer_steps.append(delay)
                                self.spike_rows.append(dest)
                                self.spike_cols.append(source)
                else:   # nonspiking chemical synapse
                    for source in self.pops_and_nrns[source_pop]:
                        for dest in self.pops_and_nrns[dest_pop]:
                            self.g_max_non[dest][source] = g_max / len(self.pops_and_nrns[source_pop])
                            self.del_e[dest][source] = del_e

    def __initialize_propagation_delay__(self) -> None:
        """
        Create a buffer sized to store enough spike data for the longest synaptic propagation delay.
        :return: None
        """
        buffer_length = int(torch.max(self.spike_delays) + 1)
        self.spike_buffer = torch.zeros([buffer_length, self.num_neurons],device=self.device)

    def __set_outputs__(self) -> None:
        """
        Build the output connectivity matrices for voltage and spike monitors and apply linear maps. Generate separate
        output monitors for each neuron in a population.
        :return: None
        """
        outputs = []
        index = 0
        for out in range(len(self.network.outputs)):
            source_pop = self.network.outputs[out]['source']
            num_source_neurons = self.network.populations[source_pop]['number']
            outputs.append([])
            for num in range(num_source_neurons):
                outputs[out].append(index)
                index += 1
        self.num_outputs = index

        self.output_voltage_connectivity = torch.zeros(
            [self.num_outputs, self.num_neurons],device=self.device)  # initialize connectivity matrix
        if self.spiking:
            self.output_spike_connectivity = torch.clone(self.output_voltage_connectivity)
        self.outputs = torch.zeros(self.num_outputs, device=self.device)
        for out in range(len(self.network.outputs)):  # iterate over the connections in the network
            source_pop = self.network.outputs[out]['source']  # get the source
            for i in range(len(self.pops_and_nrns[source_pop])):
                if self.network.outputs[out]['spiking']:
                    self.output_spike_connectivity[outputs[out][i]][
                        self.pops_and_nrns[source_pop][i]] = 1.0  # set the weight in the correct source and destination
                    # self.out_linear[outputs[out][i]] = 1.0
                else:
                    self.output_voltage_connectivity[outputs[out][i]][
                        self.pops_and_nrns[source_pop][i]] = 1.0  # set the weight in the correct source and destination

    def reset(self, u=None, theta=None, b_gate=None, c_gate=None) -> None:
        if u is None:
            self.u = torch.clone(self.u_0)
            self.u_last = torch.clone(self.u_0)
        else:
            self.u = torch.clone(u)
            self.u_last = torch.clone(u)
        if self.spiking:
            if theta is None:
                self.theta = torch.clone(self.theta_0)
                self.theta_last = torch.clone(self.theta_0)
            else:
                self.theta = torch.clone(theta)
                self.theta_last = torch.clone(theta)
        if self.gated:
            if b_gate is None:
                self.b_gate = torch.clone(self.b_gate_0)
                self.b_gate_last = torch.clone(self.b_gate_0)
            else:
                self.b_gate = torch.clone(b_gate)
                self.b_gate_last = torch.clone(b_gate)
            if c_gate is None:
                self.c_gate = torch.clone(self.c_gate_0)
                self.c_gate_last = torch.clone(self.c_gate_0)
            else:
                self.c_gate = torch.clone(c_gate)
                self.c_gate_last = torch.clone(c_gate)

    def __forward_pass__(self, inputs) -> Any:
        self.u_last = torch.clone(self.u)
        i_app = torch.matmul(self.input_connectivity, inputs)  # Apply external current sources to their destinations
        g_syn = torch.clamp(torch.minimum(self.g_max_non * self.u_last / self.R, self.g_max_non),min=0)
        if self.spiking:
            self.theta_last = torch.clone(self.theta)
            self.g_spike = self.g_spike * (1 - self.time_factor_synapse)
            g_syn += self.g_spike
        i_syn = torch.sum(g_syn * self.del_e, 1) - self.u_last * torch.sum(g_syn, 1)
        if self.electrical:
            i_syn += (torch.sum(self.g_electrical * self.u_last, 1) - self.u_last * torch.sum(self.g_electrical, 1))
        if self.electrical_rectified:
            # create mask
            mask = (self.u_last.reshape(-1,1)-self.u_last).transpose(0,1) > 0
            masked_g = mask * self.g_rectified
            diag_masked = masked_g + masked_g.transpose(0,1) - torch.diag(masked_g.diagonal())
            i_syn += torch.sum(diag_masked * self.u_last, 1) - self.u_last * torch.sum(diag_masked, 1)
        if self.gated:
            a_inf = 1 / (1 + self.k_a * torch.exp(self.slope_a*(self.e_a-self.u_last)))
            b_inf = 1 / (1 + self.k_b * torch.exp(self.slope_b*(self.e_b-self.u_last)))
            c_inf = 1 / (1 + self.k_c * torch.exp(self.slope_c*(self.e_c-self.u_last)))

            tau_b = self.tau_max_b * b_inf * torch.sqrt(self.k_b*torch.exp(self.slope_b*(self.e_b-self.u_last)))
            tau_c = self.tau_max_c * c_inf * torch.sqrt(self.k_c*torch.exp(self.slope_c*(self.e_c-self.u_last)))

            self.b_gate_last = torch.clone(self.b_gate)
            self.c_gate_last = torch.clone(self.c_gate)

            self.b_gate = self.b_gate_last + self.dt * ((b_inf - self.b_gate_last) / tau_b)
            self.c_gate = self.c_gate_last + self.dt * ((c_inf - self.c_gate_last) / tau_c)

            i_ion = self.g_ion*(a_inf**self.pow_a)*(self.b_gate**self.pow_b)*(self.c_gate**self.pow_c)*(self.e_ion-self.u_last)
            i_gated = torch.sum(i_ion, 0)

            self.u = self.u_last + self.time_factor_membrane * (-self.g_m * self.u_last + self.i_b + i_syn + i_app + i_gated)  # Update membrane potential
        else:
            self.u = self.u_last + self.time_factor_membrane * (-self.g_m * self.u_last + self.i_b + i_syn + i_app)  # Update membrane potential
        if self.spiking:
            self.theta = self.theta_last + self.time_factor_threshold * (-self.theta_last + self.theta_0 + self.m * self.u_last)  # Update the firing thresholds
            self.spikes = torch.sign(torch.clamp(self.theta - self.u,max=0))  # Compute which neurons have spiked

            # New stuff with delay
            if self.delay:
                self.spike_buffer = torch.roll(self.spike_buffer, 1, 0)   # Shift buffer entries down
                self.spike_buffer[0, :] = self.spikes    # Replace row 0 with the current spike data
                # Update a matrix with all of the appropriately delayed spike values
                self.delayed_spikes[self.spike_rows, self.spike_cols] = self.spike_buffer[self.buffer_steps, self.buffer_nrns]

                self.g_spike = torch.maximum(self.g_spike, (-self.delayed_spikes) * self.g_max_spike)  # Update the conductance of connections which spiked
            else:
                self.g_spike = torch.maximum(self.g_spike, (-self.spikes) * self.g_max_spike)  # Update the conductance of connections which spiked
            self.u = self.u * (self.spikes + 1)  # Reset the membrane voltages of neurons which spiked
        self.outputs = torch.matmul(self.output_voltage_connectivity, self.u)
        if self.spiking:
            self.outputs += torch.matmul(self.output_spike_connectivity, -self.spikes)

        return self.outputs


"""
########################################################################################################################
PYTORCH SPARSE
"""
class SNS_Sparse(__Backend__):
    """
    Simulation backend based in PyTorch Sparse. In future versions different options will be toggled automatically,
    but for now are implemented as boolean flags.

    :param network: Network which will be compiled to PyTorch Sparse.
    :type network:  sns_toolbox.design.networks.Network
    :param device:  Device network will be stored on, default is 'cuda' (GPU).
    :type device:   str, optional
    """
    def __init__(self,network: Network,device: str = 'cuda',**kwargs):
        if device != 'cpu':
            if not torch.cuda.is_available():
                warnings.warn('CUDA Device Unavailable. Using CPU Instead')
                device = 'cpu'
        self.device = device
        super().__init__(network,**kwargs)

    def __initialize_vectors_and_matrices__(self) -> None:
        """
        Initialize all of the vectors and matrices needed for all of the neural states and parameters. That includes the
        following: U, ULast, Spikes, Cm, Gm, Ibias, Theta0, Theta, ThetaLast, m, TauTheta.
        :return:    None
        """
        self.u = torch.zeros(self.num_neurons,device=self.device)
        self.u_0 = torch.zeros(self.num_neurons, device=self.device)
        self.u_last = torch.zeros(self.num_neurons,device=self.device)
        self.c_m = torch.zeros(self.num_neurons,device=self.device)
        self.g_m = torch.zeros(self.num_neurons,device=self.device)
        self.i_b = torch.sparse_coo_tensor(size=(1,self.num_neurons),device=self.device)
        if self.spiking:
            self.spikes = torch.sparse_coo_tensor(size=(1, self.num_neurons), device=self.device)
            self.theta_0 = torch.zeros(self.num_neurons,device=self.device)
            self.theta = torch.zeros(self.num_neurons,device=self.device)
            self.theta_last = torch.zeros(self.num_neurons,device=self.device)
            self.m = torch.sparse_coo_tensor(size=(1,self.num_neurons),device=self.device)
            self.tau_theta = torch.zeros(self.num_neurons,device=self.device)

        self.g_max_non = torch.sparse_coo_tensor(size=(self.num_neurons,self.num_neurons),device=self.device)
        self.del_e = torch.sparse_coo_tensor(size=(self.num_neurons, self.num_neurons), device=self.device)
        if self.spiking:
            self.g_max_spike = torch.sparse_coo_tensor(size=(self.num_neurons,self.num_neurons),device=self.device)
            self.g_spike = torch.sparse_coo_tensor(size=(self.num_neurons,self.num_neurons),device=self.device)
            self.tau_syn = torch.ones([self.num_neurons, self.num_neurons],device=self.device)
            if self.delay:
                self.spike_delays = torch.sparse_coo_tensor(size=(self.num_neurons,self.num_neurons),device=self.device)
                self.spike_rows = []
                self.spike_cols = []
                self.buffer_steps = []
                self.buffer_nrns = []
                self.delayed_spikes = torch.sparse_coo_tensor(size=(self.num_neurons,self.num_neurons),device=self.device)
        if self.electrical:
            self.g_electrical = torch.sparse_coo_tensor(size=(self.num_neurons, self.num_neurons), device=self.device)
        if self.electrical_rectified:
            self.g_rectified = torch.sparse_coo_tensor(size=(self.num_neurons, self.num_neurons), device=self.device)
        if self.gated:
            # Channel params
            self.g_ion = torch.sparse_coo_tensor(size=(self.num_channels, self.num_neurons),device=self.device)
            self.e_ion = torch.sparse_coo_tensor(size=(self.num_channels, self.num_neurons),device=self.device)
            # A gate params
            self.pow_a = torch.sparse_coo_tensor(size=(self.num_channels, self.num_neurons),device=self.device)
            self.slope_a = torch.sparse_coo_tensor(size=(self.num_channels, self.num_neurons),device=self.device)
            self.k_a = torch.ones([self.num_channels, self.num_neurons],device=self.device)
            self.e_a = torch.sparse_coo_tensor(size=(self.num_channels, self.num_neurons),device=self.device)
            # B gate params
            self.pow_b = torch.sparse_coo_tensor(size=(self.num_channels, self.num_neurons),device=self.device)
            self.slope_b = torch.sparse_coo_tensor(size=(self.num_channels, self.num_neurons),device=self.device)
            self.k_b = torch.ones([self.num_channels, self.num_neurons],device=self.device)
            self.e_b = torch.sparse_coo_tensor(size=(self.num_channels, self.num_neurons),device=self.device)
            self.tau_max_b = torch.ones([self.num_channels, self.num_neurons],device=self.device)
            # C gate params
            self.pow_c = torch.sparse_coo_tensor(size=(self.num_channels, self.num_neurons),device=self.device)
            self.slope_c = torch.sparse_coo_tensor(size=(self.num_channels, self.num_neurons),device=self.device)
            self.k_c = torch.ones([self.num_channels, self.num_neurons],device=self.device)
            self.e_c = torch.sparse_coo_tensor(size=(self.num_channels, self.num_neurons),device=self.device)
            self.tau_max_c = torch.ones([self.num_channels, self.num_neurons],device=self.device)

            self.b_gate = torch.sparse_coo_tensor(size=(self.num_channels, self.num_neurons),device=self.device)
            self.b_gate_last = torch.sparse_coo_tensor(size=(self.num_channels, self.num_neurons),device=self.device)
            self.b_gate_0 = torch.sparse_coo_tensor(size=(self.num_channels, self.num_neurons), device=self.device)
            self.c_gate = torch.sparse_coo_tensor(size=(self.num_channels, self.num_neurons),device=self.device)
            self.c_gate_last = torch.sparse_coo_tensor(size=(self.num_channels, self.num_neurons),device=self.device)
            self.c_gate_0 = torch.sparse_coo_tensor(size=(self.num_channels, self.num_neurons), device=self.device)

        self.pops_and_nrns = []
        index = 0
        for pop in range(len(self.network.populations)):
            num_neurons = self.network.populations[pop]['number']  # find the number of neurons in the population
            self.pops_and_nrns.append([])
            for num in range(num_neurons):
                self.pops_and_nrns[pop].append(index)
                index += 1

    def __set_neurons__(self) -> None:
        """
        Iterate over all populations in the network, and set the corresponding neural parameters for each neuron in the
        network: Cm, Gm, Ibias, ULast, U, Theta0, ThetaLast, Theta, TauTheta, m.
        :return:
        """
        index = 0
        for pop in range(len(self.network.populations)):
            num_neurons = self.network.populations[pop]['number']  # find the number of neurons in the population
            initial_value = self.network.populations[pop]['initial_value']
            for num in range(num_neurons):  # for each neuron, copy the parameters over
                self.c_m[index] = self.network.populations[pop]['type'].params['membrane_capacitance']
                self.g_m[index] = self.network.populations[pop]['type'].params['membrane_conductance']

                self.i_b = self.i_b.to_dense()
                self.i_b[0,index] = self.network.populations[pop]['type'].params['bias']
                self.i_b = self.i_b.to_sparse()

                if hasattr(initial_value, '__iter__'):
                    self.u_last[index] = initial_value[num]
                elif initial_value is None:
                    self.u_last[index] = 0.0
                else:
                    self.u_last[index] = initial_value

                if self.spiking:
                    if isinstance(self.network.populations[pop]['type'], SpikingNeuron):  # if the neuron is spiking, copy more
                        self.theta_0[index] = self.network.populations[pop]['type'].params['threshold_initial_value']

                        self.m = self.m.to_dense()
                        self.m[0,index] = self.network.populations[pop]['type'].params['threshold_proportionality_constant']
                        self.m = self.m.to_sparse()

                        self.tau_theta[index] = self.network.populations[pop]['type'].params['threshold_time_constant']
                    else:  # otherwise, set to the special values for NonSpiking
                        self.theta_0[index] = torch.finfo(self.theta_0[index].dtype).max

                        self.m = self.m.to_dense()
                        self.m[0,index] = 0
                        self.m = self.m.to_sparse()

                        self.tau_theta[index] = 1
                if self.gated:
                    if isinstance(self.network.populations[pop]['type'], NonSpikingNeuronWithGatedChannels):
                        # Channel params
                        self.g_ion = self.g_ion.to_dense()
                        self.g_ion[:, index] = self.network.populations[pop]['type'].params['Gion']
                        self.g_ion = self.g_ion.to_sparse()

                        self.e_ion = self.e_ion.to_dense()
                        self.e_ion[:, index] = self.network.populations[pop]['type'].params['Eion']
                        self.e_ion = self.e_ion.to_sparse()

                        # A gate params
                        self.pow_a = self.pow_a.to_dense()
                        self.pow_a[:, index] = self.network.populations[pop]['type'].params['paramsA']['pow']
                        self.pow_a = self.pow_a.to_sparse()

                        self.slope_a = self.slope_a.to_dense()
                        self.slope_a[:, index] = self.network.populations[pop]['type'].params['paramsA']['slope']
                        self.slope_a = self.slope_a.to_sparse()

                        self.k_a[:, index] = self.network.populations[pop]['type'].params['paramsA']['k']

                        self.e_a = self.e_a.to_dense()
                        self.e_a[:, index] = self.network.populations[pop]['type'].params['paramsA']['reversal']
                        self.e_a = self.e_a.to_sparse()

                        # B gate params
                        self.pow_b = self.pow_b.to_dense()
                        self.pow_b[:, index] = self.network.populations[pop]['type'].params['paramsB']['pow']
                        self.pow_b = self.pow_b.to_sparse()

                        self.slope_b = self.slope_b.to_dense()
                        self.slope_b[:, index] = self.network.populations[pop]['type'].params['paramsB']['slope']
                        self.slope_b = self.slope_b.to_sparse()

                        self.k_b[:, index] = self.network.populations[pop]['type'].params['paramsB']['k']

                        self.e_b = self.e_b.to_dense()
                        self.e_b[:, index] = self.network.populations[pop]['type'].params['paramsB']['reversal']
                        self.e_b = self.e_b.to_sparse()

                        self.tau_max_b[:, index] = self.network.populations[pop]['type'].params['paramsB']['TauMax']

                        # C gate params
                        self.pow_c = self.pow_c.to_dense()
                        self.pow_c[:, index] = self.network.populations[pop]['type'].params['paramsC']['pow']
                        self.pow_c = self.pow_c.to_sparse()

                        self.slope_c = self.slope_c.to_dense()
                        self.slope_c[:, index] = self.network.populations[pop]['type'].params['paramsC']['slope']
                        self.slope_c = self.slope_c.to_sparse()

                        self.k_c[:, index] = self.network.populations[pop]['type'].params['paramsC']['k']

                        self.e_c = self.e_c.to_dense()
                        self.e_c[:, index] = self.network.populations[pop]['type'].params['paramsC']['reversal']
                        self.e_c = self.e_c.to_sparse()

                        self.tau_max_c[:, index] = self.network.populations[pop]['type'].params['paramsC']['TauMax']

                        self.b_gate_last = self.b_gate_last.to_dense()
                        self.b_gate_last[:, index] = 1 / (1 + self.k_b[:, index] * np.exp(self.slope_b.to_dense()[:, index]*(self.u_last[index]-self.e_b.to_dense()[:, index])))
                        self.b_gate_last = self.b_gate_last.to_sparse()

                        self.c_gate_last = self.c_gate_last.to_dense()
                        self.c_gate_last[:, index] = 1 / (1 + self.k_c[:, index] * np.exp(self.slope_c.to_dense()[:, index]*(self.u_last[index]-self.e_c.to_dense()[:, index])))
                        self.c_gate_last = self.c_gate_last.to_sparse()

                index += 1
        self.u = self.u_last.clone()
        if self.spiking:
            self.theta = self.theta_0.clone()
            self.theta_last = self.theta_0.clone()
        if self.gated:
            self.b_gate = torch.clone(self.b_gate_last)
            self.b_gate_0 = torch.clone(self.b_gate_last)
            self.c_gate = torch.clone(self.c_gate_last)
            self.c_gate_0 = torch.clone(self.c_gate_last)

    def __set_inputs__(self) -> None:
        """
        Build the input connection matrix, and apply linear mapping coefficients.
        :return:    None
        """
        self.input_connectivity = torch.sparse_coo_tensor(size=(self.num_neurons, self.network.get_num_inputs_actual()),device=self.device)  # initialize connectivity matrix
        index = 0
        for inp in range(self.network.get_num_inputs()):  # iterate over the connections in the network
            size = self.network.inputs[inp]['size']
            dest_pop = self.network.inputs[inp]['destination']  # get the destination

            self.input_connectivity = self.input_connectivity.to_dense()
            if size == 1:
                for dest in self.pops_and_nrns[dest_pop]:
                    self.input_connectivity[dest][inp] = 1.0  # set the weight in the correct source and destination
                index += 1
            else:
                for dest in self.pops_and_nrns[dest_pop]:
                    self.input_connectivity[dest][index] = 1.0
                    index += 1
            self.input_connectivity = self.input_connectivity.to_sparse()

    def __set_connections__(self) -> None:
        """
        Build the synaptic parameter matrices. Interpret connectivity patterns between populations into individual
        synapses.
        :return: None
        """
        for syn in range(len(self.network.connections)):
            source_pop = self.network.connections[syn]['source']
            dest_pop = self.network.connections[syn]['destination']
            g_max = self.network.connections[syn]['params']['max_conductance']
            if self.network.connections[syn]['params']['electrical'] is False:  # chemical connection
                del_e = self.network.connections[syn]['params']['reversal_potential']

            if self.network.connections[syn]['params']['pattern']:  # pattern connection
                pop_size = len(self.pops_and_nrns[source_pop])
                source_index = self.pops_and_nrns[source_pop][0]
                dest_index = self.pops_and_nrns[dest_pop][0]

                if self.network.connections[syn]['params']['spiking']:
                    tau_s = self.network.connections[syn]['params']['synapticTimeConstant']

                    self.g_max_spike = self.g_max_spike.to_dense()
                    self.g_max_spike[dest_index:dest_index + pop_size,source_index:source_index + pop_size] = torch.from_numpy(g_max)
                    self.g_max_spike = self.g_max_spike.to_sparse()

                    self.del_e = self.del_e.to_dense()
                    self.del_e[dest_index:dest_index + pop_size,source_index:source_index + pop_size] = torch.from_numpy(del_e)
                    self.del_e = self.del_e.to_sparse()

                    self.tau_syn[dest_index:dest_index+pop_size,source_index:source_index+pop_size] = torch.from_numpy(tau_s)

                    if self.delay:
                        delay = self.network.connections[syn]['params']['synapticTransmissionDelay']
                        self.spike_delays = self.spike_delays.to_dense()
                        self.spike_delays[dest_index:dest_index+pop_size,source_index:source_index+pop_size] = torch.from_numpy(delay)
                        self.spike_delays = self.spike_delays.to_sparse()

                        for source in self.pops_and_nrns[source_pop]:
                            for dest in self.pops_and_nrns[dest_pop]:
                                self.buffer_nrns.append(source)
                                self.buffer_steps.append(delay)
                                self.spike_rows.append(dest)
                                self.spike_cols.append(source)
                else:
                    self.g_max_non = self.g_max_non.to_dense()
                    self.g_max_non[dest_index:dest_index+pop_size,source_index:source_index+pop_size] = torch.from_numpy(g_max)
                    self.g_max_non = self.g_max_non.to_sparse()

                    self.del_e = self.del_e.to_dense()
                    self.del_e[dest_index:dest_index+pop_size,source_index:source_index+pop_size] = torch.from_numpy(del_e)
                    self.del_e = self.del_e.to_sparse()
            elif self.network.connections[syn]['params']['electrical']:  # electrical connection
                for source in self.pops_and_nrns[source_pop]:
                    for dest in self.pops_and_nrns[dest_pop]:
                        if self.network.connections[syn]['params']['rectified']:  # rectified
                            self.g_rectified = self.g_rectified.to_dense()
                            self.g_rectified[dest][source] = g_max / len(self.pops_and_nrns[source_pop])
                            self.g_rectified = self.g_rectified.to_sparse()
                        else:
                            self.g_electrical = self.g_electrical.to_dense()
                            self.g_electrical[dest][source] = g_max / len(self.pops_and_nrns[source_pop])
                            self.g_electrical[source][dest] = g_max / len(self.pops_and_nrns[source_pop])
                            self.g_electrical = self.g_electrical.to_sparse()
            else:   # chemical connections
                if self.network.connections[syn]['params']['spiking']:  # spiking chemical synapse
                    tau_s = self.network.connections[syn]['params']['synapticTimeConstant']
                    if self.delay:
                        delay = self.network.connections[syn]['params']['synapticTransmissionDelay']
                    for source in self.pops_and_nrns[source_pop]:
                        for dest in self.pops_and_nrns[dest_pop]:
                            self.g_max_spike = self.g_max_spike.to_dense()
                            self.g_max_spike[dest][source] = g_max / len(self.pops_and_nrns[source_pop])
                            self.g_max_spike = self.g_max_spike.to_sparse()

                            self.del_e = self.del_e.to_dense()
                            self.del_e[dest][source] = del_e
                            self.del_e = self.del_e.to_sparse()

                            self.tau_syn[dest][source] = tau_s

                            if self.delay:
                                self.spike_delays = self.spike_delays.to_dense()
                                self.spike_delays[dest][source] = delay
                                self.spike_delays = self.spike_delays.to_sparse()

                                self.buffer_nrns.append(source)
                                self.buffer_steps.append(delay)
                                self.spike_rows.append(dest)
                                self.spike_cols.append(source)
                else:   # non-spiking chemical synapse
                    for source in self.pops_and_nrns[source_pop]:
                        for dest in self.pops_and_nrns[dest_pop]:
                            self.g_max_non = self.g_max_non.to_dense()
                            self.g_max_non[dest][source] = g_max / len(self.pops_and_nrns[source_pop])
                            self.g_max_non = self.g_max_non.to_sparse()

                            self.del_e = self.del_e.to_dense()
                            self.del_e[dest][source] = del_e
                            self.del_e = self.del_e.to_sparse()

    def __initialize_propagation_delay__(self) -> None:
        """
        Create a buffer sized to store enough spike data for the longest synaptic propagation delay.
        :return: None
        """
        self.spike_delays = self.spike_delays.to_dense()
        buffer_length = int(torch.max(self.spike_delays) + 1)
        self.spike_delays = self.spike_delays.to_sparse()

        self.spike_buffer = torch.sparse_coo_tensor(size=(buffer_length,self.num_neurons),device=self.device)

    def __set_outputs__(self) -> None:
        """
        Build the output connectivity matrices for voltage and spike monitors and apply linear maps. Generate separate
        output monitors for each neuron in a population.
        :return: None
        """
        outputs = []
        index = 0
        for out in range(len(self.network.outputs)):
            source_pop = self.network.outputs[out]['source']
            num_source_neurons = self.network.populations[source_pop]['number']
            outputs.append([])
            for num in range(num_source_neurons):
                outputs[out].append(index)
                index += 1
        self.num_outputs = index

        self.output_voltage_connectivity = torch.sparse_coo_tensor(size=(self.num_outputs, self.num_neurons),device=self.device)  # initialize connectivity matrix
        if self.spiking:
            self.output_spike_connectivity = torch.clone(self.output_voltage_connectivity)
        self.outputs = torch.sparse_coo_tensor(size=(1, self.num_outputs), device=self.device)

        for out in range(len(self.network.outputs)):  # iterate over the connections in the network
            source_pop = self.network.outputs[out]['source']  # get the source
            for i in range(len(self.pops_and_nrns[source_pop])):
                if self.network.outputs[out]['spiking']:
                    self.output_spike_connectivity = self.output_spike_connectivity.to_dense()
                    self.output_spike_connectivity[outputs[out][i]][
                        self.pops_and_nrns[source_pop][i]] = 1.0  # set the weight in the correct source and destination
                    self.output_spike_connectivity = self.output_spike_connectivity.to_sparse()
                else:
                    self.output_voltage_connectivity = self.output_voltage_connectivity.to_dense()
                    self.output_voltage_connectivity[outputs[out][i]][
                        self.pops_and_nrns[source_pop][i]] = 1.0  # set the weight in the correct source and destination
                    self.output_voltage_connectivity = self.output_voltage_connectivity.to_sparse()

    def reset(self, u=None, theta=None, b_gate=None, c_gate=None) -> None:
        if u is None:
            self.u = torch.clone(self.u_0)
            self.u_last = torch.clone(self.u_0)
        else:
            self.u = torch.clone(u)
            self.u_last = torch.clone(u)
        if self.spiking:
            if theta is None:
                self.theta = torch.clone(self.theta_0)
                self.theta_last = torch.clone(self.theta_0)
            else:
                self.theta = torch.clone(theta)
                self.theta_last = torch.clone(theta)
        if self.gated:
            if b_gate is None:
                self.b_gate = torch.clone(self.b_gate_0)
                self.b_gate_last = torch.clone(self.b_gate_0)
            else:
                self.b_gate = torch.clone(b_gate)
                self.b_gate_last = torch.clone(b_gate)
            if c_gate is None:
                self.c_gate = torch.clone(self.c_gate_0)
                self.c_gate_last = torch.clone(self.c_gate_0)
            else:
                self.c_gate = torch.clone(c_gate)
                self.c_gate_last = torch.clone(c_gate)

    def __forward_pass__(self, inputs) -> Any:
        self.u_last = torch.clone(self.u)

        i_app = torch.matmul(self.input_connectivity, inputs)  # Apply external current sources to their destinations
        i_app = i_app.to_sparse()

        g_syn = torch.clamp(torch.minimum(self.g_max_non.to_dense() * self.u_last / self.R, self.g_max_non.to_dense()),min=0)
        g_syn = g_syn.to_sparse()

        if self.spiking:
            self.theta_last = torch.clone(self.theta)
            self.g_spike = self.g_spike.to_dense() * (1 - self.time_factor_synapse)
            self.g_spike = self.g_spike.to_sparse()

            g_syn += self.g_spike

        if g_syn._nnz() > 0:
            i_syn = torch.sparse.sum(g_syn * self.del_e, 1) - (self.u_last * torch.sum(g_syn.to_dense(), 1)).to_sparse()
        else:
            i_syn = torch.sparse.sum(g_syn * self.del_e) - self.u_last * torch.sparse.sum(g_syn)
        if self.electrical:
            i_syn += (torch.sum(self.g_electrical.to_dense() * self.u_last, 1).to_sparse() -
                      (self.u_last * torch.sum(self.g_electrical.to_dense(), 1)).to_sparse())
        if self.electrical_rectified:
            # create mask
            mask = (self.u_last.reshape(-1, 1) - self.u_last).transpose(0, 1) > 0
            masked_g = mask * self.g_rectified.to_dense()
            diag_masked = masked_g + masked_g.transpose(0, 1) - torch.diag(masked_g.diagonal())
            i_syn += torch.sum(diag_masked * self.u_last, 1).to_sparse() - (self.u_last * torch.sum(diag_masked, 1)).to_sparse()
        if self.gated:
            a_inf = (1 / (1 + self.k_a * torch.exp(self.slope_a.to_dense()*(self.e_a.to_dense()-self.u_last)))).to_sparse()
            b_inf = (1 / (1 + self.k_b * torch.exp(self.slope_b.to_dense()*(self.e_b.to_dense()-self.u_last)))).to_sparse()
            c_inf = (1 / (1 + self.k_c * torch.exp(self.slope_c.to_dense()*(self.e_c.to_dense()-self.u_last)))).to_sparse()

            tau_b = (self.tau_max_b * b_inf.to_dense() * torch.sqrt(self.k_b*torch.exp(self.slope_b.to_dense()*(self.e_b.to_dense()-self.u_last)))).to_sparse()
            tau_c = (self.tau_max_c * c_inf.to_dense() * torch.sqrt(self.k_c*torch.exp(self.slope_c.to_dense()*(self.e_c.to_dense()-self.u_last)))).to_sparse()

            self.b_gate_last = torch.clone(self.b_gate)
            self.c_gate_last = torch.clone(self.c_gate)

            self.b_gate = (self.b_gate_last.to_dense() + self.dt * ((b_inf - self.b_gate_last).to_dense() / tau_b.to_dense())).to_sparse()
            self.c_gate = (self.c_gate_last.to_dense() + self.dt * ((c_inf - self.c_gate_last).to_dense() / tau_c.to_dense())).to_sparse()

            i_ion = (self.g_ion.to_dense()*(a_inf.to_dense()**self.pow_a.to_dense())*(self.b_gate.to_dense()**self.pow_b.to_dense())*(self.c_gate.to_dense()**self.pow_c.to_dense())*(self.e_ion.to_dense()-self.u_last)).to_sparse()
            i_gated = torch.sum(i_ion.to_dense(), 0).to_sparse()

            self.u = self.u_last + self.time_factor_membrane * (-self.g_m * self.u_last + (self.i_b.to_dense())[0,:] + i_syn + i_app + i_gated)  # Update membrane potential
        else:
            self.u = self.u_last + self.time_factor_membrane * (-self.g_m * self.u_last + (self.i_b.to_dense())[0,:] + i_syn + i_app)  # Update membrane potential
        if self.spiking:
            self.theta = self.theta_last + self.time_factor_threshold * (-self.theta_last + self.theta_0 + (self.m.to_dense())[0,:] * self.u_last)  # Update the firing thresholds

            self.spikes = torch.sign(torch.clamp(self.theta - self.u,max=0))  # Compute which neurons have spiked
            self.spikes = self.spikes.to_sparse()

            if self.delay:
                # New stuff with delay
                self.spike_buffer = self.spike_buffer.to_dense()
                self.spike_buffer = torch.roll(self.spike_buffer, 1, 0)   # Shift buffer entries down
                self.spike_buffer[0, :] = self.spikes.to_dense()    # Replace row 0 with the current spike data
                self.spike_buffer = self.spike_buffer.to_sparse()

                # Update a matrix with all of the appropriately delayed spike values
                self.delayed_spikes = self.delayed_spikes.to_dense()
                self.delayed_spikes[self.spike_rows, self.spike_cols] = (self.spike_buffer.to_dense())[self.buffer_steps, self.buffer_nrns]
                self.delayed_spikes = self.delayed_spikes.to_sparse()

                self.g_spike = torch.maximum(self.g_spike.to_dense(), ((-self.delayed_spikes) * self.g_max_spike).to_dense())  # Update the conductance of connections which spiked
            else:
                self.g_spike = torch.maximum(self.g_spike.to_dense(), (-self.spikes.to_dense()) * self.g_max_spike.to_dense())  # Update the conductance of connections which spiked
            self.g_spike = self.g_spike.to_sparse()
            self.u = self.u * (self.spikes.to_dense() + 1)  # Reset the membrane voltages of neurons which spiked
        self.outputs = torch.matmul(self.output_voltage_connectivity, self.u)
        if self.spiking:
            self.outputs += torch.matmul(self.output_spike_connectivity, -self.spikes.to_dense())

        return self.outputs


"""
########################################################################################################################
MANUAL BACKEND

Simulating the network using numpy vectors and matrices.
Note that this is not sparse, so memory may explode for large networks
"""
class SNS_Manual(__Backend__):
    """
    Simulation backend based in Numpy, but computes neural and synapses states in loops. Primarily for comparison, but
    could be useful for giant networks that don't fit in memory. In future versions different options will be toggled
    automatically, but for now are implemented as boolean flags.

    :param network: Network which will be compiled.
    :type network:  sns_toolbox.design.networks.Network
    """
    def __init__(self,network: Network,**kwargs):
        super().__init__(network,**kwargs)

    def __initialize_vectors_and_matrices__(self) -> None:
        """
        Initialize all of the vectors and matrices needed for all of the neural states and parameters. That includes the
        following: U, ULast, Spikes, Cm, Gm, Ibias, Theta0, Theta, ThetaLast, m, TauTheta.
        :return:    None
        """
        self.u = np.zeros(self.num_neurons)
        self.u_0 = np.zeros(self.num_neurons)
        self.u_last = np.zeros(self.num_neurons)
        self.c_m = np.zeros(self.num_neurons)
        self.g_m = np.zeros(self.num_neurons)
        self.i_b = np.zeros(self.num_neurons)
        if self.spiking:
            self.spikes = np.zeros(self.num_neurons)
            self.theta_0 = np.zeros(self.num_neurons)
            self.theta = np.zeros(self.num_neurons)
            self.theta_last = np.zeros(self.num_neurons)
            self.m = np.zeros(self.num_neurons)
            self.tau_theta = np.zeros(self.num_neurons)
        if self.gated:
            # Channel params
            self.g_ion = np.zeros([self.num_channels, self.num_neurons])
            self.e_ion = np.zeros([self.num_channels, self.num_neurons])
            # A gate params
            self.pow_a = np.zeros([self.num_channels, self.num_neurons])
            self.slope_a = np.zeros([self.num_channels, self.num_neurons])
            self.k_a = np.zeros([self.num_channels, self.num_neurons])+1
            self.e_a = np.zeros([self.num_channels, self.num_neurons])
            # B gate params
            self.pow_b = np.zeros([self.num_channels, self.num_neurons])
            self.slope_b = np.zeros([self.num_channels, self.num_neurons])
            self.k_b = np.zeros([self.num_channels, self.num_neurons])+1
            self.e_b = np.zeros([self.num_channels, self.num_neurons])
            self.tau_max_b = np.zeros([self.num_channels, self.num_neurons])+1
            # C gate params
            self.pow_c = np.zeros([self.num_channels, self.num_neurons])
            self.slope_c = np.zeros([self.num_channels, self.num_neurons])
            self.k_c = np.zeros([self.num_channels, self.num_neurons])+1
            self.e_c = np.zeros([self.num_channels, self.num_neurons])
            self.tau_max_c = np.zeros([self.num_channels, self.num_neurons])+1

            self.b_gate = np.zeros([self.num_channels, self.num_neurons])
            self.b_gate_0 = np.zeros([self.num_channels, self.num_neurons])
            self.b_gate_last = np.zeros([self.num_channels, self.num_neurons])
            self.c_gate = np.zeros([self.num_channels, self.num_neurons])
            self.c_gate_0 = np.zeros([self.num_channels, self.num_neurons])
            self.c_gate_last = np.zeros([self.num_channels, self.num_neurons])

        self.incoming_synapses = []
        for i in range(self.num_neurons):
            self.incoming_synapses.append([])

        self.pops_and_nrns = []
        index = 0
        for pop in range(len(self.network.populations)):
            num_neurons = self.network.populations[pop]['number']  # find the number of neurons in the population
            self.pops_and_nrns.append([])
            for num in range(num_neurons):
                self.pops_and_nrns[pop].append(index)
                index += 1

    def __set_neurons__(self) -> None:
        """
        Iterate over all populations in the network, and set the corresponding neural parameters for each neuron in the
        network: Cm, Gm, Ibias, ULast, U, Theta0, ThetaLast, Theta, TauTheta, m.
        :return:
        """
        index = 0
        for pop in range(len(self.network.populations)):
            num_neurons = self.network.populations[pop]['number']  # find the number of neurons in the population
            initial_value = self.network.populations[pop]['initial_value']
            for num in range(num_neurons):  # for each neuron, copy the parameters over
                self.c_m[index] = self.network.populations[pop]['type'].params['membrane_capacitance']
                self.g_m[index] = self.network.populations[pop]['type'].params['membrane_conductance']
                self.i_b[index] = self.network.populations[pop]['type'].params['bias']
                if hasattr(initial_value, '__iter__'):
                    self.u_last[index] = initial_value[num]
                elif initial_value is None:
                    self.u_last[index] = 0.0
                else:
                    self.u_last[index] = initial_value
                if self.spiking:
                    if isinstance(self.network.populations[pop]['type'], SpikingNeuron):  # if the neuron is spiking, copy more
                        self.theta_0[index] = self.network.populations[pop]['type'].params['threshold_initial_value']
                        self.m[index] = self.network.populations[pop]['type'].params['threshold_proportionality_constant']
                        self.tau_theta[index] = self.network.populations[pop]['type'].params['threshold_time_constant']
                    else:  # otherwise, set to the special values for NonSpiking
                        self.theta_0[index] = sys.float_info.max
                        self.m[index] = 0
                        self.tau_theta[index] = 1
                if self.gated:
                    if isinstance(self.network.populations[pop]['type'], NonSpikingNeuronWithGatedChannels):
                        # Channel params
                        self.g_ion[:, index] = self.network.populations[pop]['type'].params['Gion']
                        self.e_ion[:, index] = self.network.populations[pop]['type'].params['Eion']
                        # A gate params
                        self.pow_a[:, index] = self.network.populations[pop]['type'].params['paramsA']['pow']
                        self.slope_a[:, index] = self.network.populations[pop]['type'].params['paramsA']['slope']
                        self.k_a[:, index] = self.network.populations[pop]['type'].params['paramsA']['k']
                        self.e_a[:, index] = self.network.populations[pop]['type'].params['paramsA']['reversal']
                        # B gate params
                        self.pow_b[:, index] = self.network.populations[pop]['type'].params['paramsB']['pow']
                        self.slope_b[:, index] = self.network.populations[pop]['type'].params['paramsB']['slope']
                        self.k_b[:, index] = self.network.populations[pop]['type'].params['paramsB']['k']
                        self.e_b[:, index] = self.network.populations[pop]['type'].params['paramsB']['reversal']
                        self.tau_max_b[:, index] = self.network.populations[pop]['type'].params['paramsB']['TauMax']
                        # C gate params
                        self.pow_c[:, index] = self.network.populations[pop]['type'].params['paramsC']['pow']
                        self.slope_c[:, index] = self.network.populations[pop]['type'].params['paramsC']['slope']
                        self.k_c[:, index] = self.network.populations[pop]['type'].params['paramsC']['k']
                        self.e_c[:, index] = self.network.populations[pop]['type'].params['paramsC']['reversal']
                        self.tau_max_c[:, index] = self.network.populations[pop]['type'].params['paramsC']['TauMax']

                        self.b_gate_last[:, index] = 1 / (1 + self.k_b[:, index] * np.exp(self.slope_b[:, index]*(self.u_last[index]-self.e_b[:, index])))
                        self.c_gate_last[:, index] = 1 / (1 + self.k_c[:, index] * np.exp(self.slope_c[:, index]*(self.u_last[index]-self.e_c[:, index])))
                index += 1
        self.u = np.copy(self.u_last)
        if self.spiking:
            self.theta = np.copy(self.theta_0)
            self.theta_last = np.copy(self.theta_0)
        if self.gated:
            self.b_gate = np.copy(self.b_gate_last)
            self.b_gate_0 = np.copy(self.b_gate_last)
            self.c_gate = np.copy(self.c_gate_last)
            self.c_gate_0 = np.copy(self.c_gate_last)

    def __set_inputs__(self) -> None:
        """
        Build the input connection matrix, and apply linear mapping coefficients.
        :return:    None
        """
        self.input_connectivity = np.zeros([self.num_neurons, self.network.get_num_inputs_actual()])  # initialize connectivity matrix
        index = 0
        for inp in range(self.network.get_num_inputs()):  # iterate over the connections in the network
            size = self.network.inputs[inp]['size']
            dest_pop = self.network.inputs[inp]['destination']  # get the destination
            if size == 1:
                for dest in self.pops_and_nrns[dest_pop]:
                    self.input_connectivity[dest][inp] = 1.0  # set the weight in the correct source and destination
                index += 1
            else:
                for dest in self.pops_and_nrns[dest_pop]:
                    self.input_connectivity[dest][index] = 1.0
                    index += 1

    def __set_connections__(self) -> None:
        """
        Build the synaptic parameter matrices. Interpret connectivity patterns between populations into individual
        synapses.
        :return: None
        """
        for syn in range(len(self.network.connections)):
            source_pop = self.network.connections[syn]['source']
            dest_pop = self.network.connections[syn]['destination']
            g_max = self.network.connections[syn]['params']['max_conductance']
            if self.network.connections[syn]['params']['electrical'] is False:  # electrical connection
                del_e = self.network.connections[syn]['params']['reversal_potential']

            if self.network.connections[syn]['params']['pattern']:  # pattern connection
                pop_size = len(self.pops_and_nrns[source_pop])
                source_index = self.pops_and_nrns[source_pop][0]
                dest_index = self.pops_and_nrns[dest_pop][0]
                if self.network.connections[syn]['params']['spiking']:
                    tau_s = self.network.connections[syn]['params']['synapticTimeConstant']
                    if self.delay:
                        delay = self.network.connections[syn]['params']['synapticTransmissionDelay']

                    for dest in range(pop_size):
                        for source in range(pop_size):
                            g_syn = g_max[dest,source]
                            rev = del_e[dest,source]
                            time_factor_syn = self.dt/tau_s[dest,source]
                            if self.delay:
                                buffer = np.zeros(delay[dest, source])
                                self.incoming_synapses[dest+dest_index].append([source+source_index, True, False, g_syn, rev, 0, time_factor_syn, buffer])
                            else:
                                self.incoming_synapses[dest + dest_index].append([source + source_index, True, False, g_syn, rev, 0, time_factor_syn])
                else:
                    for dest in range(pop_size):
                        for source in range(pop_size):
                            g_syn = g_max[dest, source]
                            rev = del_e[dest, source]

                            self.incoming_synapses[dest + dest_index].append([source + source_index, False, False, g_syn, rev, 0])
            elif self.network.connections[syn]['params']['electrical']:  # electrical connection
                for dest in self.pops_and_nrns[dest_pop]:
                    for source in self.pops_and_nrns[source_pop]:
                        g_syn = g_max / len(self.pops_and_nrns[source_pop])
                        if self.network.connections[syn]['params']['rectified']:  # rectified
                            self.incoming_synapses[dest].append([source, False, True, g_syn, True, source, dest])
                            self.incoming_synapses[source].append([dest, False, True, g_syn, True, source, dest])
                        else:
                            self.incoming_synapses[dest].append([source, False, True, g_syn, False])
                            self.incoming_synapses[source].append([dest, False, True, g_syn, False])
            else:   # chemical connection
                if self.network.connections[syn]['params']['spiking']:
                    tau_s = self.network.connections[syn]['params']['synapticTimeConstant']
                    if self.delay:
                        delay = self.network.connections[syn]['params']['synapticTransmissionDelay']
                    for dest in self.pops_and_nrns[dest_pop]:
                        for source in self.pops_and_nrns[source_pop]:
                            g_syn = g_max / len(self.pops_and_nrns[source_pop])
                            if self.delay:
                                buffer = np.zeros(delay+1)
                                self.incoming_synapses[dest].append([source, True, False, g_syn, del_e, 0, self.dt/tau_s,buffer])
                            else:
                                self.incoming_synapses[dest].append([source, True, False, g_syn, del_e, 0, self.dt / tau_s])
                else:
                    for dest in self.pops_and_nrns[dest_pop]:
                        for source in self.pops_and_nrns[source_pop]:
                            g_syn = g_max / len(self.pops_and_nrns[source_pop])
                            self.incoming_synapses[dest].append([source,False, False, g_syn,del_e,0])

    def __calculate_time_factors__(self) -> None:
        """
        Precompute the time factors for the membrane voltage, firing threshold, and spiking synapses.
        :return: None
        """
        self.time_factor_membrane = self.dt / (self.c_m/self.g_m)
        if self.spiking:
            self.time_factor_threshold = self.dt / self.tau_theta

    def __initialize_propagation_delay__(self) -> None:
        """
        Create a buffer sized to store enough spike data for the longest synaptic propagation delay.
        :return: None
        """
        pass

    def __set_outputs__(self) -> None:
        """
        Build the output connectivity matrices for voltage and spike monitors and apply linear maps. Generate separate
        output monitors for each neuron in a population.
        :return: None
        """
        outputs = []
        index = 0
        for out in range(len(self.network.outputs)):
            source_pop = self.network.outputs[out]['source']
            num_source_neurons = self.network.populations[source_pop]['number']
            outputs.append([])
            for num in range(num_source_neurons):
                outputs[out].append(index)
                index += 1
        self.num_outputs = index

        self.output_voltage_connectivity = np.zeros([self.num_outputs, self.num_neurons])  # initialize connectivity matrix
        if self.spiking:
            self.output_spike_connectivity = np.copy(self.output_voltage_connectivity)
        self.outputs = np.zeros(self.num_outputs)
        for out in range(len(self.network.outputs)):  # iterate over the connections in the network
            source_pop = self.network.outputs[out]['source']  # get the source
            for i in range(len(self.pops_and_nrns[source_pop])):
                if self.network.outputs[out]['spiking']:
                    self.output_spike_connectivity[outputs[out][i]][
                        self.pops_and_nrns[source_pop][i]] = 1.0  # set the weight in the correct source and destination
                    # self.out_linear[outputs[out][i]] = 1.0
                else:
                    self.output_voltage_connectivity[outputs[out][i]][
                        self.pops_and_nrns[source_pop][i]] = 1.0  # set the weight in the correct source and destination

    def __debug_print__(self) -> None:
        """
        Print the values for every vector/matrix which will be used in the forward computation.
        :return: None
        """
        print('Input Connectivity:')
        print(self.input_connectivity)
        print('Output Voltage Connectivity')
        print(self.output_voltage_connectivity)
        if self.spiking:
            print('Output Spike Connectivity:')
            print(self.output_spike_connectivity)
        print('u:')
        print(self.u)
        print('u_last:')
        print(self.u_last)
        if self.spiking:
            print('theta_0:')
            print(self.theta_0)
            print('ThetaLast:')
            print(self.theta_last)
            print('Theta')
            print(self.theta)
        if self.gated:
            print('Number of Channels:')
            print(self.num_channels)
            print('Ionic Conductance:')
            print(self.g_ion)
            print('Ionic Reversal Potentials:')
            print(self.e_ion)
            print('A Gate Parameters:')
            print('Power:')
            print(self.pow_a)
            print('Slope:')
            print(self.slope_a)
            print('K:')
            print(self.k_a)
            print('Reversal Potential:')
            print(self.e_a)
            print('B Gate Parameters:')
            print('Power:')
            print(self.pow_b)
            print('Slope:')
            print(self.slope_b)
            print('K:')
            print(self.k_b)
            print('Reversal Potential:')
            print(self.e_b)
            print('Tau Max:')
            print(self.tau_max_b)
            print('B:')
            print(self.b_gate)
            print('B_last:')
            print(self.b_gate_last)
            print('C Gate Parameters:')
            print('Power:')
            print(self.pow_c)
            print('Slope:')
            print(self.slope_c)
            print('K:')
            print(self.k_c)
            print('Reversal Potential:')
            print(self.e_c)
            print('Tau Max:')
            print(self.tau_max_c)
            print('B:')
            print(self.c_gate)
            print('B_last:')
            print(self.c_gate_last)

    def reset(self, u=None, theta=None, b_gate=None, c_gate=None) -> None:
        if u is None:
            self.u = np.copy(self.u_0)
            self.u_last = np.copy(self.u_0)
        else:
            self.u = np.copy(u)
            self.u_last = np.copy(u)
        if self.spiking:
            if theta is None:
                self.theta = np.copy(self.theta_0)
                self.theta_last = np.copy(self.theta_0)
            else:
                self.theta = np.copy(theta)
                self.theta_last = np.copy(theta)
        if self.gated:
            if b_gate is None:
                self.b_gate = np.copy(self.b_gate_0)
                self.b_gate_last = np.copy(self.b_gate_0)
            else:
                self.b_gate = np.copy(b_gate)
                self.b_gate_last = np.copy(b_gate)
            if c_gate is None:
                self.c_gate = np.copy(self.c_gate_0)
                self.c_gate_last = np.copy(self.c_gate_0)
            else:
                self.c_gate = np.copy(c_gate)
                self.c_gate_last = np.copy(c_gate)

    def __forward_pass__(self, inputs) -> Any:
        self.u_last = np.copy(self.u)
        if self.spiking:
            self.theta_last = np.copy(self.theta)
        if self.gated:
            self.b_gate_last = np.copy(self.b_gate)
            self.c_gate_last = np.copy(self.c_gate)

        i_app = np.matmul(self.input_connectivity, inputs)  # Apply external current sources to their destinations

        for nrn in range(self.num_neurons):
            i_syn = 0
            for syn in range(len(self.incoming_synapses[nrn])):
                neuron_src = self.incoming_synapses[nrn][syn]
                if neuron_src[1]:  # if spiking
                    neuron_src[5] = neuron_src[5] * (1-neuron_src[6])
                    i_syn += neuron_src[5] * (neuron_src[4] - self.u_last[nrn])
                elif neuron_src[2]: # if electrical
                    if neuron_src[4]:   # if rectified
                        if self.u_last[neuron_src[5]] > self.u_last[neuron_src[6]]:
                            i_syn += neuron_src[3] * (self.u_last[neuron_src[0]] - self.u_last[nrn])
                    else:
                        i_syn += neuron_src[3] * (self.u_last[neuron_src[0]] - self.u_last[nrn])
                else:   # if chemical
                    neuron_src[5] = np.maximum(0, np.minimum(neuron_src[3] * self.u_last[neuron_src[0]] / self.R, neuron_src[3]))
                    i_syn += neuron_src[5] * (neuron_src[4] - self.u_last[nrn])
            i_gated = 0
            if self.gated:
                a_inf = 1 / (1 + self.k_a[:,nrn] * np.exp(self.slope_a[:,nrn] * (self.e_a[:,nrn] - self.u_last[nrn])))
                b_inf = 1 / (1 + self.k_b[:,nrn] * np.exp(self.slope_b[:,nrn] * (self.e_b[:,nrn] - self.u_last[nrn])))
                c_inf = 1 / (1 + self.k_c[:,nrn] * np.exp(self.slope_c[:,nrn] * (self.e_c[:,nrn] - self.u_last[nrn])))

                tau_b = self.tau_max_b[:,nrn] * b_inf * np.sqrt(self.k_b[:,nrn] * np.exp(self.slope_b[:,nrn] * (self.e_b[:,nrn] - self.u_last[nrn])))
                tau_c = self.tau_max_c[:,nrn] * c_inf * np.sqrt(self.k_c[:,nrn] * np.exp(self.slope_c[:,nrn] * (self.e_c[:,nrn] - self.u_last[nrn])))

                self.b_gate[:,nrn] = self.b_gate_last[:,nrn] + self.dt * ((b_inf - self.b_gate_last[:,nrn]) / tau_b)
                self.c_gate[:,nrn] = self.c_gate_last[:,nrn] + self.dt * ((c_inf - self.c_gate_last[:,nrn]) / tau_c)

                i_ion = self.g_ion[:,nrn] * (a_inf ** self.pow_a[:,nrn]) * (self.b_gate[:,nrn] ** self.pow_b[:,nrn]) * (self.c_gate[:,nrn] ** self.pow_c[:,nrn]) * (self.e_ion[:,nrn] - self.u_last[nrn])
                i_gated = np.sum(i_ion)

            self.u[nrn] = self.u_last[nrn] + self.time_factor_membrane[nrn] * (-self.g_m[nrn] * self.u_last[nrn] + self.i_b[nrn] + i_syn + i_app[nrn] + i_gated)  # Update membrane potential
            if self.spiking:
                # if self.theta_0[nrn] != sys.float_info.max:
                self.theta[nrn] = self.theta_last[nrn] + self.time_factor_threshold[nrn] * (-self.theta_last[nrn] + self.theta_0[nrn] + self.m[nrn] * self.u_last[nrn])  # Update the firing thresholds
                self.spikes[nrn] = np.sign(np.minimum(0, self.theta[nrn] - self.u[nrn]))  # Compute which neurons have spiked
        if self.spiking:
            for nrn in range(self.num_neurons):
                if self.delay:
                    # New stuff with delay
                    for syn in range(len(self.incoming_synapses[nrn])):
                        neuron_src = self.incoming_synapses[nrn][syn]
                        if neuron_src[1]:  # if spiking
                            neuron_src[7] = np.roll(neuron_src[7], 1)   # Shift buffer entries down
                            neuron_src[7][0] = self.spikes[neuron_src[0]]    # Replace row 0 with the current spike data
                            neuron_src[5] = np.maximum(neuron_src[5], (-neuron_src[7][-1]) * neuron_src[3])  # Update the conductance of connections which spiked
                self.u[nrn] = self.u[nrn] * (self.spikes[nrn] + 1)  # Reset the membrane voltages of neurons which spiked
        self.outputs = np.matmul(self.output_voltage_connectivity, self.u)
        if self.spiking:
            self.outputs += np.matmul(self.output_spike_connectivity, -self.spikes)

        return self.outputs
