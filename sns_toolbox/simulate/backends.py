"""
Simulation backends for nonspiking networks. Each of these are python-based, and are constructed using a Nonspiking
Network. They can then be run for a step, with the inputs being a vector of neural states and applied currents and the
output being the next step of neural states.
William Nourse
August 31, 2021
I've heard that you're a low-down Yankee liar
"""

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
IMPORTS
"""

from typing import Any
import numpy as np
import torch
import sys

from sns_toolbox.design.networks import Network
from sns_toolbox.design.neurons import SpikingNeuron
from sns_toolbox.design.connections import SpikingSynapse
from sns_toolbox.simulate.__utilities__ import send_vars

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
BASE CLASS
"""

class Backend:
    """
    Base-level class for all simulation backends. Each will do the following:
        - Construct a representation of a given network using the desired backend technique
        - Take in (some form of) a vector of input states and applied currents, and compute the result for the next
          timestep
    """
    def __init__(self, network: Network, dt: float = 0.1, debug: bool = False, substeps: int = 1) -> None:
        """
        Construct the backend based on the network design
        :param network: NonSpikingNetwork to serve as a design template
        :param dt:      Simulation time constant
        :param debug: Flag for printing debug information to the console
        :param substeps: Number of simulation substeps before returning an output vector
        """
        if substeps <= 0:
            raise ValueError('Substeps must be a positive integer')
        self.substeps = substeps
        self.network = network
        self.dt = dt
        self.debug = debug

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
        Initialize all of the vectors and matrices needed for all of the neural states and parameters. That includes the
        following: U, ULast, Spikes, Cm, Gm, Ibias, Theta0, Theta, ThetaLast, m, TauTheta.
        :return:    None
        """
        raise NotImplementedError

    def __set_neurons__(self) -> None:
        """
        Iterate over all populations in the network, and set the corresponding neural parameters for each neuron in the
        network: Cm, Gm, Ibias, ULast, U, Theta0, ThetaLast, Theta, TauTheta, m.
        :return:
        """
        raise NotImplementedError

    def __set_inputs__(self) -> None:
        """
        Build the input connection matrix, and apply linear mapping coefficients.
        :return:    None
        """
        raise NotImplementedError

    def __set_connections__(self) -> None:
        """
        Build the synaptic parameter matrices. Interpret connectivity patterns between populations into individual
        synapses.
        :return: None
        """
        raise NotImplementedError

    def __calculate_time_factors__(self) -> None:
        """
        Precompute the time factors for the membrane voltage, firing threshold, and spiking synapses.
        :return: None
        """
        raise NotImplementedError

    def __initialize_propagation_delay__(self) -> None:
        """
        Create a buffer sized to store enough spike data for the longest synaptic propagation delay.
        :return: None
        """
        raise NotImplementedError

    def __set_outputs__(self) -> None:
        """
        Build the output connectivity matrices for voltage and spike monitors and apply linear maps. Generate separate
        output monitors for each neuron in a population.
        :return: None
        """
        raise NotImplementedError

    def __debug_print__(self) -> None:
        """
        Print the values for every vector/matrix which will be used in the forward computation.
        :return: None
        """
        raise NotImplementedError

    def forward(self, inputs) -> Any:
        """
        Compute the next neural states based on previous neural states
        :param inputs:    Input currents into the network
        :return:          The next neural voltages
        """
        for i in range(self.substeps):
            out = self.__forward_pass__(inputs)
        return out

    def __forward_pass__(self, inputs) -> Any:
        """
        Compute the next neural states based on previous neural states in the following steps:
        Ulast = U
        ThetaLast = Theta
        MappedInputs = cubic*inputs^3 + quadratic*inputs^2 + linear*inputs + offset
        IApp = InputConnectivity X MappedInputs
        GNon = max(0, min(GMaxNon*ULast/R, GMaxNon))
        GSpike = GSpike * (1-TimeFactorSynapse)
        GSyn = GNon + GSpike
        ISyn = ColSum(GSyn*DelE) - ULast*ColSum(GSyn)
        U = ULast + TimeFactorMembrane*(-Gm*ULast + IBias + ISyn + IApp)
        Theta = ThetaLast + TimeFactorThreshold*(-ThetaLast + Theta0 + m*ULast)
        Spikes = Sign(min(0, Theta - U))
        SpikeBuffer = SpikeBuffer shifted down by 1
        SpikeBuffer[first row] = Spikes
        DelayedSpikeMatrix = SpikeBuffer[BufferSteps, BufferedNeurons]
        GSpike = max(GSpike, -DelayedSpikeMatrix*GMaxSpike)
        U = U * (Spikes + 1)
        Outputs = OutputVoltageConnectivity X U + OutputSpikeConnectivity X (-Spikes)
        MappedOutputs = cubic*Outputs^3 + quadratic*Outputs^2 + linear*Outputs + offset
        :param inputs:    Input currents into the network
        :return:          The next neural voltages
        """
        raise NotImplementedError

"""
########################################################################################################################
NUMPY BACKEND

Simulating the network using numpy vectors and matrices.
Note that this is not sparse, so memory may explode for large networks
"""

class SNS_Numpy(Backend):
    def __init__(self,network: Network,**kwargs):
        super().__init__(network,**kwargs)

    def __initialize_vectors_and_matrices__(self) -> None:
        """
        Initialize all of the vectors and matrices needed for all of the neural states and parameters. That includes the
        following: U, ULast, Spikes, Cm, Gm, Ibias, Theta0, Theta, ThetaLast, m, TauTheta.
        :return:    None
        """
        self.u = np.zeros(self.num_neurons)
        self.u_last = np.zeros(self.num_neurons)
        self.spikes = np.zeros(self.num_neurons)
        self.c_m = np.zeros(self.num_neurons)
        self.g_m = np.zeros(self.num_neurons)
        self.i_b = np.zeros(self.num_neurons)
        self.theta_0 = np.zeros(self.num_neurons)
        self.theta = np.zeros(self.num_neurons)
        self.theta_last = np.zeros(self.num_neurons)
        self.m = np.zeros(self.num_neurons)
        self.tau_theta = np.zeros(self.num_neurons)

        self.g_max_non = np.zeros([self.num_neurons, self.num_neurons])
        self.g_max_spike = np.zeros([self.num_neurons, self.num_neurons])
        self.g_spike = np.zeros([self.num_neurons, self.num_neurons])
        self.del_e = np.zeros([self.num_neurons, self.num_neurons])
        self.tau_syn = np.zeros([self.num_neurons, self.num_neurons]) + 1
        self.spike_delays = np.zeros([self.num_neurons, self.num_neurons])
        self.spike_rows = []
        self.spike_cols = []
        self.buffer_steps = []
        self.buffer_nrns = []
        self.spike_delay_inds = np.zeros([self.num_neurons ** 2])
        self.delayed_spikes = np.zeros([self.num_neurons, self.num_neurons])

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
            u_last = 0.0
            for num in range(num_neurons):  # for each neuron, copy the parameters over
                self.c_m[index] = self.network.populations[pop]['type'].params['membrane_capacitance']
                self.g_m[index] = self.network.populations[pop]['type'].params['membrane_conductance']
                self.i_b[index] = self.network.populations[pop]['type'].params['bias']
                if hasattr(initial_value, '__iter__'):
                    self.u_last[index] = initial_value[num]
                else:
                    self.u_last[index] = initial_value
                # TODO: Change this part to accommodate new populations
                if isinstance(self.network.populations[pop]['type'], SpikingNeuron):  # if the neuron is spiking, copy more
                    self.theta_0[index] = self.network.populations[pop]['type'].params['threshold_initial_value']
                    # u_last += self.network.populations[pop]['type'].params['threshold_initial_value'] / num_neurons
                    self.m[index] = self.network.populations[pop]['type'].params['threshold_proportionality_constant']
                    self.tau_theta[index] = self.network.populations[pop]['type'].params['threshold_time_constant']
                else:  # otherwise, set to the special values for NonSpiking
                    self.theta_0[index] = sys.float_info.max
                    self.m[index] = 0
                    self.tau_theta[index] = 1
                    # u_last += self.R / num_neurons
                index += 1
        self.u = np.copy(self.u_last)
        self.theta = np.copy(self.theta_0)
        self.theta_last = np.copy(self.theta_0)

    def __set_inputs__(self) -> None:
        """
        Build the input connection matrix, and apply linear mapping coefficients.
        :return:    None
        """
        self.input_connectivity = np.zeros([self.num_neurons, self.num_inputs])  # initialize connectivity matrix
        self.in_offset = np.zeros(self.num_inputs)
        self.in_linear = np.zeros(self.num_inputs)
        self.in_quad = np.zeros(self.num_inputs)
        self.in_cubic = np.zeros(self.num_inputs)
        self.inputs_mapped = np.zeros(self.num_inputs)
        for inp in range(self.network.get_num_inputs()):  # iterate over the connections in the network
            self.in_offset[inp] = self.network.inputs[inp]['offset']
            self.in_linear[inp] = self.network.inputs[inp]['linear']
            self.in_quad[inp] = self.network.inputs[inp]['quadratic']
            self.in_cubic[inp] = self.network.inputs[inp]['cubic']
            dest_pop = self.network.inputs[inp]['destination']  # get the destination
            for dest in self.pops_and_nrns[dest_pop]:
                self.input_connectivity[dest][inp] = 1.0  # set the weight in the correct source and destination

    def __set_connections__(self) -> None:
        """
        Build the synaptic parameter matrices. Interpret connectivity patterns between populations into individual
        synapses.
        :return: None
        """
        # TODO: Patterned Connections
        for syn in range(len(self.network.connections)):
            source_pop = self.network.connections[syn]['source']
            dest_pop = self.network.connections[syn]['destination']
            g_max = self.network.connections[syn]['type'].params['max_conductance']
            del_e = self.network.connections[syn]['type'].params['relative_reversal_potential']

            if isinstance(self.network.connections[syn]['type'], SpikingSynapse):
                tau_s = self.network.connections[syn]['type'].params['synapticTimeConstant']
                delay = self.network.connections[syn]['type'].params['synapticTransmissionDelay']
                for source in self.pops_and_nrns[source_pop]:
                    for dest in self.pops_and_nrns[dest_pop]:
                        self.g_max_spike[dest][source] = g_max / len(self.pops_and_nrns[source_pop])
                        self.del_e[dest][source] = del_e
                        self.tau_syn[dest][source] = tau_s
                        self.spike_delays[dest][source] = delay
                        self.buffer_nrns.append(source)
                        self.buffer_steps.append(delay)
                        self.spike_rows.append(dest)
                        self.spike_cols.append(source)
            else:
                for source in self.pops_and_nrns[source_pop]:
                    for dest in self.pops_and_nrns[dest_pop]:
                        self.g_max_non[dest][source] = g_max / len(self.pops_and_nrns[source_pop])
                        self.del_e[dest][source] = del_e

    def __calculate_time_factors__(self) -> None:
        """
        Precompute the time factors for the membrane voltage, firing threshold, and spiking synapses.
        :return: None
        """
        self.time_factor_membrane = self.dt / self.c_m
        self.time_factor_threshold = self.dt / self.tau_theta
        self.time_factor_synapse = self.dt / self.tau_syn

    def __initialize_propagation_delay__(self) -> None:
        """
        Create a buffer sized to store enough spike data for the longest synaptic propagation delay.
        :return: None
        """
        buffer_length = int(np.max(self.spike_delays) + 1)
        self.spike_buffer = np.zeros([buffer_length, self.num_neurons])

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

        self.output_voltage_connectivity = np.zeros(
            [self.num_outputs, self.num_neurons])  # initialize connectivity matrix
        self.output_spike_connectivity = np.copy(self.output_voltage_connectivity)
        self.out_offset = np.zeros(self.num_outputs)
        self.out_linear = np.zeros(self.num_outputs)
        self.out_quad = np.zeros(self.num_outputs)
        self.out_cubic = np.zeros(self.num_outputs)
        self.outputs_raw = np.zeros(self.num_outputs)
        for out in range(len(self.network.outputs)):  # iterate over the connections in the network
            source_pop = self.network.outputs[out]['source']  # get the source
            if self.network.outputs[out]['spiking']:
                self.out_linear[out] = 1.0
            else:
                self.out_offset[out] = self.network.outputs[out]['offset']
                self.out_linear[out] = self.network.outputs[out]['linear']
                self.out_quad[out] = self.network.outputs[out]['quadratic']
                self.out_cubic[out] = self.network.outputs[out]['cubic']
            for i in range(len(self.pops_and_nrns[source_pop])):
                if self.network.outputs[out]['spiking']:
                    self.output_spike_connectivity[outputs[out][i]][
                        self.pops_and_nrns[source_pop][i]] = 1.0  # set the weight in the correct source and destination
                    self.out_linear[outputs[out][i]] = 1.0
                else:
                    self.output_voltage_connectivity[outputs[out][i]][
                        self.pops_and_nrns[source_pop][i]] = 1.0  # set the weight in the correct source and destination
                    self.out_offset[outputs[out][i]] = self.network.outputs[out]['offset']
                    self.out_linear[outputs[out][i]] = self.network.outputs[out]['linear']
                    self.out_quad[outputs[out][i]] = self.network.outputs[out]['quadratic']
                    self.out_cubic[outputs[out][i]] = self.network.outputs[out]['cubic']

    def __debug_print__(self) -> None:
        """
        Print the values for every vector/matrix which will be used in the forward computation.
        :return: None
        """
        print('Input Connectivity:')
        print(self.input_connectivity)
        print('g_max_non:')
        print(self.g_max_non)
        print('GmaxSpike:')
        print(self.g_max_spike)
        print('del_e:')
        print(self.del_e)
        print('Output Voltage Connectivity')
        print(self.output_voltage_connectivity)
        print('Output Spike Connectivity:')
        print(self.output_spike_connectivity)
        print('u:')
        print(self.u)
        print('u_last:')
        print(self.u_last)
        print('theta_0:')
        print(self.theta_0)
        print('ThetaLast:')
        print(self.theta_last)
        print('Theta')
        print(self.theta)

    def __forward_pass__(self, inputs) -> Any:
        self.u_last = np.copy(self.u)
        self.theta_last = np.copy(self.theta)
        self.inputs_mapped = self.in_cubic*(inputs**3) + self.in_quad*(inputs**2) + self.in_linear*inputs + self.in_offset
        i_app = np.matmul(self.input_connectivity, self.inputs_mapped)  # Apply external current sources to their destinations
        g_non = np.maximum(0, np.minimum(self.g_max_non * self.u_last / self.R, self.g_max_non))
        self.g_spike = self.g_spike * (1 - self.time_factor_synapse)
        g_syn = g_non + self.g_spike
        i_syn = np.sum(g_syn * self.del_e, axis=1) - self.u_last * np.sum(g_syn, axis=1)
        self.u = self.u_last + self.time_factor_membrane * (-self.g_m * self.u_last + self.i_b + i_syn + i_app)  # Update membrane potential
        self.theta = self.theta_last + self.time_factor_threshold * (-self.theta_last + self.theta_0 + self.m * self.u_last)  # Update the firing thresholds
        self.spikes = np.sign(np.minimum(0, self.theta - self.u))  # Compute which neurons have spiked

        # New stuff with delay
        self.spike_buffer = np.roll(self.spike_buffer, 1, axis=0)   # Shift buffer entries down
        self.spike_buffer[0, :] = self.spikes    # Replace row 0 with the current spike data
        # Update a matrix with all of the appropriately delayed spike values
        self.delayed_spikes[self.spike_rows, self.spike_cols] = self.spike_buffer[self.buffer_steps, self.buffer_nrns]

        self.g_spike = np.maximum(self.g_spike, (-self.delayed_spikes) * self.g_max_spike)  # Update the conductance of connections which spiked
        self.u = self.u * (self.spikes + 1)  # Reset the membrane voltages of neurons which spiked
        self.outputs_raw = np.matmul(self.output_voltage_connectivity, self.u) + np.matmul(self.output_spike_connectivity, -self.spikes)

        return self.out_cubic*(self.outputs_raw**3) + self.out_quad*(self.outputs_raw**2)\
            + self.out_linear*self.outputs_raw + self.out_offset

# TODO: Redo with inheritance
class SNS_Numpy_No_Delay(Backend):
    def __init__(self,network: Network,**kwargs):
        super().__init__(network,**kwargs)

        """Neurons"""
        if self.debug:
            print('BUILDING NEURONS')
        # Initialize the vectors
        self.u = np.zeros(self.num_neurons)
        self.u_last = np.zeros(self.num_neurons)
        self.spikes = np.zeros(self.num_neurons)
        c_m = np.zeros(self.num_neurons)
        self.g_m = np.zeros(self.num_neurons)
        self.i_b = np.zeros(self.num_neurons)
        self.theta_0 = np.zeros(self.num_neurons)
        self.theta = np.zeros(self.num_neurons)
        self.theta_last = np.zeros(self.num_neurons)
        self.m = np.zeros(self.num_neurons)
        tau_theta = np.zeros(self.num_neurons)

        # iterate over the populations in the network
        pops_and_nrns = []
        index = 0
        for pop in range(len(network.populations)):
            num_neurons = network.populations[pop]['number'] # find the number of neurons in the population
            pops_and_nrns.append([])
            u_last = 0.0
            for num in range(num_neurons):   # for each neuron, copy the parameters over
                c_m[index] = network.populations[pop]['type'].params['membrane_capacitance']
                self.g_m[index] = network.populations[pop]['type'].params['membrane_conductance']
                self.i_b[index] = network.populations[pop]['type'].params['bias']
                self.u_last[index] = u_last
                if isinstance(network.populations[pop]['type'],SpikingNeuron):  # if the neuron is spiking, copy more
                    self.theta_0[index] = network.populations[pop]['type'].params['threshold_initial_value']
                    u_last += network.populations[pop]['type'].params['threshold_initial_value']/num_neurons
                    self.m[index] = network.populations[pop]['type'].params['threshold_proportionality_constant']
                    tau_theta[index] = network.populations[pop]['type'].params['threshold_time_constant']
                else:   # otherwise, set to the special values for NonSpiking
                    self.theta_0[index] = sys.float_info.max
                    self.m[index] = 0
                    tau_theta[index] = 1
                    u_last += self.R/num_neurons
                pops_and_nrns[pop].append(index)
                index += 1
        self.u = np.copy(self.u_last)
        # set the derived vectors
        self.time_factor_membrane = self.dt / c_m
        self.time_factor_threshold = self.dt / tau_theta
        self.theta = np.copy(self.theta_0)
        self.theta_last = np.copy(self.theta_0)

        """Inputs"""
        if self.debug:
            print('BUILDING INPUTS')
        self.input_connectivity = np.zeros([self.num_neurons, self.num_inputs])  # initialize connectivity matrix
        self.in_offset = np.zeros(self.num_inputs)
        self.in_linear = np.zeros(self.num_inputs)
        self.in_quad = np.zeros(self.num_inputs)
        self.in_cubic = np.zeros(self.num_inputs)
        self.inputs_mapped = np.zeros(self.num_inputs)
        for inp in range(network.get_num_inputs()):  # iterate over the connections in the network
            self.in_offset[inp] = network.inputs[inp]['offset']
            self.in_linear[inp] = network.inputs[inp]['linear']
            self.in_quad[inp] = network.inputs[inp]['quadratic']
            self.in_cubic[inp] = network.inputs[inp]['cubic']
            dest_pop = network.inputs[inp]['destination']  # get the destination
            for dest in pops_and_nrns[dest_pop]:
                self.input_connectivity[dest][inp] = 1.0  # set the weight in the correct source and destination

        """Synapses"""
        if self.debug:
            print('BUILDING SYNAPSES')
        # initialize the matrices
        self.g_max_non = np.zeros([self.num_neurons, self.num_neurons])
        self.g_max_spike = np.zeros([self.num_neurons, self.num_neurons])
        self.g_spike = np.zeros([self.num_neurons, self.num_neurons])
        self.del_e = np.zeros([self.num_neurons, self.num_neurons])
        self.tau_syn = np.zeros([self.num_neurons, self.num_neurons]) + 1

        # iterate over the connections in the network
        for syn in range(len(network.connections)):
            source_pop = network.connections[syn]['source']
            dest_pop = network.connections[syn]['destination']
            g_max = network.connections[syn]['type'].params['max_conductance']
            del_e = network.connections[syn]['type'].params['relative_reversal_potential']

            if isinstance(network.connections[syn]['type'], SpikingSynapse):
                tau_s = network.connections[syn]['type'].params['synapticTimeConstant']
                for source in pops_and_nrns[source_pop]:
                    for dest in pops_and_nrns[dest_pop]:
                        self.g_max_spike[dest][source] = g_max / len(pops_and_nrns[source_pop])
                        self.del_e[dest][source] = del_e
                        self.tau_syn[dest][source] = tau_s
            else:
                for source in pops_and_nrns[source_pop]:
                    for dest in pops_and_nrns[dest_pop]:
                        self.g_max_non[dest][source] = g_max / len(pops_and_nrns[source_pop])
                        self.del_e[dest][source] = del_e
        self.time_factor_synapse = self.dt / self.tau_syn

        """Outputs"""
        if self.debug:
            print('BUILDING OUTPUTS')
        # Figure out how many outputs there actually are, since an output has as many elements as its input population
        outputs = []
        index = 0
        for out in range(len(network.outputs)):
            source_pop = network.outputs[out]['source']
            num_source_neurons = network.populations[source_pop]['number']
            outputs.append([])
            for num in range(num_source_neurons):
                outputs[out].append(index)
                index += 1
        self.num_outputs = index

        self.output_voltage_connectivity = np.zeros([self.num_outputs, self.num_neurons])  # initialize connectivity matrix
        self.output_spike_connectivity = np.copy(self.output_voltage_connectivity)
        self.out_offset = np.zeros(self.num_outputs)
        self.out_linear = np.zeros(self.num_outputs)
        self.out_quad = np.zeros(self.num_outputs)
        self.out_cubic = np.zeros(self.num_outputs)
        self.outputs_raw = np.zeros(self.num_outputs)
        for out in range(len(network.outputs)):  # iterate over the connections in the network
            source_pop = network.outputs[out]['source']  # get the source
            if network.outputs[out]['spiking']:
                self.out_linear[out] = 1.0
            else:
                self.out_offset[out] = network.outputs[out]['offset']
                self.out_linear[out] = network.outputs[out]['linear']
                self.out_quad[out] = network.outputs[out]['quadratic']
                self.out_cubic[out] = network.outputs[out]['cubic']
            for i in range(len(pops_and_nrns[source_pop])):
                if network.outputs[out]['spiking']:
                    self.output_spike_connectivity[outputs[out][i]][pops_and_nrns[source_pop][i]] = 1.0  # set the weight in the correct source and destination
                    self.out_linear[outputs[out][i]] = 1.0
                else:
                    self.output_voltage_connectivity[outputs[out][i]][pops_and_nrns[source_pop][i]] = 1.0  # set the weight in the correct source and destination
                    self.out_offset[outputs[out][i]] = network.outputs[out]['offset']
                    self.out_linear[outputs[out][i]] = network.outputs[out]['linear']
                    self.out_quad[outputs[out][i]] = network.outputs[out]['quadratic']
                    self.out_cubic[outputs[out][i]] = network.outputs[out]['cubic']
        if self.debug:
            print('Input Connectivity:')
            print(self.input_connectivity)
            print('g_max_non:')
            print(self.g_max_non)
            print('GmaxSpike:')
            print(self.g_max_spike)
            print('del_e:')
            print(self.del_e)
            print('Output Voltage Connectivity')
            print(self.output_voltage_connectivity)
            print('Output Spike Connectivity:')
            print(self.output_spike_connectivity)
            print('u:')
            print(self.u)
            print('u_last:')
            print(self.u_last)
            print('theta_0:')
            print(self.theta_0)
            print('ThetaLast:')
            print(self.theta_last)
            print('Theta')
            print(self.theta)
            print('\nDONE BUILDING')

    def forward(self, inputs) -> Any:
        self.u_last = np.copy(self.u)
        self.theta_last = np.copy(self.theta)
        self.inputs_mapped = self.in_cubic*(inputs**3) + self.in_quad*(inputs**2) + self.in_linear*inputs + self.in_offset
        i_app = np.matmul(self.input_connectivity, self.inputs_mapped)  # Apply external current sources to their destinations
        g_non = np.maximum(0, np.minimum(self.g_max_non * self.u_last / self.R, self.g_max_non))
        self.g_spike = self.g_spike * (1 - self.time_factor_synapse)
        g_syn = g_non + self.g_spike
        i_syn = np.sum(g_syn * self.del_e, axis=1) - self.u_last * np.sum(g_syn, axis=1)
        self.u = self.u_last + self.time_factor_membrane * (-self.g_m * self.u_last + self.i_b + i_syn + i_app)  # Update membrane potential
        self.theta = self.theta_last + self.time_factor_threshold * (-self.theta_last + self.theta_0 + self.m * self.u_last)  # Update the firing thresholds
        self.spikes = np.sign(np.minimum(0, self.theta - self.u))  # Compute which neurons have spiked
        self.g_spike = np.maximum(self.g_spike, (-self.spikes) * self.g_max_spike)  # Update the conductance of connections which spiked
        self.u = self.u * (self.spikes + 1)  # Reset the membrane voltages of neurons which spiked
        self.outputs_raw = np.matmul(self.output_voltage_connectivity, self.u) + np.matmul(self.output_spike_connectivity, -self.spikes)

        return self.out_cubic*(self.outputs_raw**3) + self.out_quad*(self.outputs_raw**2)\
            + self.out_linear*self.outputs_raw + self.out_offset

# TODO: Redo with inheritance
class SNS_Numpy_Non_Spiking(Backend):
    def __init__(self,network: Network,**kwargs):
        super().__init__(network,**kwargs)

        """Neurons"""
        if self.debug:
            print('BUILDING NEURONS')
        # Initialize the vectors
        self.u = np.zeros(self.num_neurons)
        self.u_last = np.zeros(self.num_neurons)
        c_m = np.zeros(self.num_neurons)
        self.g_m = np.zeros(self.num_neurons)
        self.i_b = np.zeros(self.num_neurons)

        # iterate over the populations in the network
        pops_and_nrns = []
        index = 0
        for pop in range(len(network.populations)):
            num_neurons = network.populations[pop]['number'] # find the number of neurons in the population
            pops_and_nrns.append([])
            u_last = 0.0
            for num in range(num_neurons):   # for each neuron, copy the parameters over
                c_m[index] = network.populations[pop]['type'].params['membrane_capacitance']
                self.g_m[index] = network.populations[pop]['type'].params['membrane_conductance']
                self.i_b[index] = network.populations[pop]['type'].params['bias']
                self.u_last[index] = u_last
                pops_and_nrns[pop].append(index)
                index += 1
        self.u = np.copy(self.u_last)
        # set the derived vectors
        self.time_factor_membrane = self.dt / c_m

        """Inputs"""
        if self.debug:
            print('BUILDING INPUTS')
        self.input_connectivity = np.zeros([self.num_neurons, self.num_inputs])  # initialize connectivity matrix
        self.in_offset = np.zeros(self.num_inputs)
        self.in_linear = np.zeros(self.num_inputs)
        self.in_quad = np.zeros(self.num_inputs)
        self.in_cubic = np.zeros(self.num_inputs)
        self.inputs_mapped = np.zeros(self.num_inputs)
        for inp in range(network.get_num_inputs()):  # iterate over the connections in the network
            self.in_offset[inp] = network.inputs[inp]['offset']
            self.in_linear[inp] = network.inputs[inp]['linear']
            self.in_quad[inp] = network.inputs[inp]['quadratic']
            self.in_cubic[inp] = network.inputs[inp]['cubic']
            dest_pop = network.inputs[inp]['destination']  # get the destination
            for dest in pops_and_nrns[dest_pop]:
                self.input_connectivity[dest][inp] = 1.0  # set the weight in the correct source and destination

        """Synapses"""
        if self.debug:
            print('BUILDING SYNAPSES')
        # initialize the matrices
        self.g_max = np.zeros([self.num_neurons, self.num_neurons])
        self.del_e = np.zeros([self.num_neurons, self.num_neurons])

        # iterate over the connections in the network
        for syn in range(len(network.connections)):
            source_pop = network.connections[syn]['source']
            dest_pop = network.connections[syn]['destination']
            g_max = network.connections[syn]['type'].params['max_conductance']
            del_e = network.connections[syn]['type'].params['relative_reversal_potential']

            for source in pops_and_nrns[source_pop]:
                for dest in pops_and_nrns[dest_pop]:
                    self.g_max[dest][source] = g_max / len(pops_and_nrns[source_pop])
                    self.del_e[dest][source] = del_e

        """Outputs"""
        if self.debug:
            print('BUILDING OUTPUTS')
        # Figure out how many outputs there actually are, since an output has as many elements as its input population
        outputs = []
        index = 0
        for out in range(len(network.outputs)):
            source_pop = network.outputs[out]['source']
            num_source_neurons = network.populations[source_pop]['number']
            outputs.append([])
            for num in range(num_source_neurons):
                outputs[out].append(index)
                index += 1
        self.num_outputs = index

        self.output_voltage_connectivity = np.zeros([self.num_outputs, self.num_neurons])  # initialize connectivity matrix
        self.out_offset = np.zeros(self.num_outputs)
        self.out_linear = np.zeros(self.num_outputs)
        self.out_quad = np.zeros(self.num_outputs)
        self.out_cubic = np.zeros(self.num_outputs)
        self.outputs_raw = np.zeros(self.num_outputs)
        for out in range(len(network.outputs)):  # iterate over the connections in the network
            source_pop = network.outputs[out]['source']  # get the source
            self.out_offset[out] = network.outputs[out]['offset']
            self.out_linear[out] = network.outputs[out]['linear']
            self.out_quad[out] = network.outputs[out]['quadratic']
            self.out_cubic[out] = network.outputs[out]['cubic']
            for i in range(len(pops_and_nrns[source_pop])):
                self.output_voltage_connectivity[outputs[out][i]][pops_and_nrns[source_pop][i]] = 1.0  # set the weight in the correct source and destination
                self.out_offset[outputs[out][i]] = network.outputs[out]['offset']
                self.out_linear[outputs[out][i]] = network.outputs[out]['linear']
                self.out_quad[outputs[out][i]] = network.outputs[out]['quadratic']
                self.out_cubic[outputs[out][i]] = network.outputs[out]['cubic']
        if self.debug:
            print('Input Connectivity:')
            print(self.input_connectivity)
            print('g_max_non:')
            print(self.g_max)
            print('del_e:')
            print(self.del_e)
            print('Output Voltage Connectivity')
            print(self.output_voltage_connectivity)
            print('u:')
            print(self.u)
            print('u_last:')
            print(self.u_last)
            print('\nDONE BUILDING')

    def forward(self, inputs) -> Any:
        self.u_last = np.copy(self.u)
        self.inputs_mapped = self.in_cubic*(inputs**3) + self.in_quad*(inputs**2) + self.in_linear*inputs + self.in_offset
        i_app = np.matmul(self.input_connectivity, self.inputs_mapped)  # Apply external current sources to their destinations
        g_syn = np.maximum(0, np.minimum(self.g_max * self.u_last / self.R, self.g_max))
        i_syn = np.sum(g_syn * self.del_e, axis=1) - self.u_last * np.sum(g_syn, axis=1)
        self.u = self.u_last + self.time_factor_membrane * (-self.g_m * self.u_last + self.i_b + i_syn + i_app)  # Update membrane potential
        self.outputs_raw = np.matmul(self.output_voltage_connectivity, self.u)

        return self.out_cubic*(self.outputs_raw**3) + self.out_quad*(self.outputs_raw**2)\
            + self.out_linear*self.outputs_raw + self.out_offset

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
PYTORCH DENSE

Simulating the network using GPU-compatible tensors.
Note that this is not sparse, so memory may explode for large networks
"""
# TODO: Redo with inheritance
class SNS_Torch(Backend):
    # TODO: Add polynomial mapping
    # TODO: Add synaptic transmission delay
    def __init__(self, network: Network,device: str = 'cuda', **kwargs):
        super().__init__(network, **kwargs)

        """Neurons"""
        if self.debug:
            print('BUILDING NEURONS')
        # Initialize the vectors
        self.u = torch.from_numpy(np.zeros(self.num_neurons))
        self.u_last = torch.from_numpy(np.zeros(self.num_neurons))
        self.spikes = torch.from_numpy(np.zeros(self.num_neurons))
        c_m = torch.from_numpy(np.zeros(self.num_neurons))
        self.g_m = torch.from_numpy(np.zeros(self.num_neurons))
        self.i_b = torch.from_numpy(np.zeros(self.num_neurons))
        self.theta_0 = torch.from_numpy(np.zeros(self.num_neurons))
        self.theta = torch.from_numpy(np.zeros(self.num_neurons))
        self.theta_last = torch.from_numpy(np.zeros(self.num_neurons))
        self.m = torch.from_numpy(np.zeros(self.num_neurons))
        tau_theta = torch.from_numpy(np.zeros(self.num_neurons))

        # iterate over the populations in the network
        pops_and_nrns = []
        index = 0
        for pop in range(len(network.populations)):
            num_neurons = network.populations[pop]['number']  # find the number of neurons in the population
            pops_and_nrns.append([])
            u_last = 0.0
            for num in range(num_neurons):  # for each neuron, copy the parameters over
                c_m[index] = network.populations[pop]['type'].params['membrane_capacitance']
                self.g_m[index] = network.populations[pop]['type'].params['membrane_conductance']
                self.i_b[index] = network.populations[pop]['type'].params['bias']
                self.u_last[index] = u_last
                if isinstance(network.populations[pop]['type'], SpikingNeuron):  # if the neuron is spiking, copy more
                    self.theta_0[index] = network.populations[pop]['type'].params['threshold_initial_value']
                    u_last += network.populations[pop]['type'].params['threshold_initial_value'] / num_neurons
                    self.m[index] = network.populations[pop]['type'].params['threshold_proportionality_constant']
                    tau_theta[index] = network.populations[pop]['type'].params['threshold_time_constant']
                else:  # otherwise, set to the special values for NonSpiking
                    self.theta_0[index] = sys.float_info.max
                    self.m[index] = 0
                    tau_theta[index] = 1
                    u_last += self.R / num_neurons
                pops_and_nrns[pop].append(index)
                index += 1
        self.u = self.u_last.clone()
        # set the derived vectors
        self.time_factor_membrane = self.dt / c_m
        self.time_factor_threshold = self.dt / tau_theta
        self.theta = self.theta_0.clone()
        self.theta_last = self.theta_0.clone()

        """Inputs"""
        if self.debug:
            print('BUILDING INPUTS')
        self.input_connectivity = torch.from_numpy(np.zeros([self.num_neurons, self.num_inputs]))  # initialize connectivity matrix
        for conn in network.inputConns:  # iterate over the connections in the network
            wt = conn['weight']  # get the weight
            source = conn['source']  # get the source
            dest_pop = conn['destination']  # get the destination
            for dest in pops_and_nrns[dest_pop]:
                self.input_connectivity[dest, source] = wt  # set the weight in the correct source and destination

        """Synapses"""
        if self.debug:
            print('BUILDING SYNAPSES')
        # initialize the matrices
        self.g_max_non = torch.from_numpy(np.zeros([self.num_neurons, self.num_neurons]))
        # self.zeros = self.g_max_non.clone()
        self.g_max_spike = torch.from_numpy(np.zeros([self.num_neurons, self.num_neurons]))
        self.g_spike = torch.from_numpy(np.zeros([self.num_neurons, self.num_neurons]))
        self.del_e = torch.from_numpy(np.zeros([self.num_neurons, self.num_neurons]))
        self.tau_syn = torch.from_numpy(np.zeros([self.num_neurons, self.num_neurons])) + 1

        # iterate over the connections in the network
        for syn in range(len(network.connections)):
            source_pop = network.connections[syn]['source']
            dest_pop = network.connections[syn]['destination']
            g_max = network.connections[syn]['type'].params['max_conductance']
            del_e = network.connections[syn]['type'].params['relative_reversal_potential']

            if isinstance(network.connections[syn]['type'], SpikingSynapse):
                tau_s = network.connections[syn]['type'].params['synapticTimeConstant']
                for source in pops_and_nrns[source_pop]:
                    for dest in pops_and_nrns[dest_pop]:
                        self.g_max_spike[dest, source] = g_max / len(pops_and_nrns[source_pop])
                        self.del_e[dest, source] = del_e
                        self.tau_syn[dest, source] = tau_s
            else:
                for source in pops_and_nrns[source_pop]:
                    for dest in pops_and_nrns[dest_pop]:
                        self.g_max_non[dest, source] = g_max / len(pops_and_nrns[source_pop])
                        self.del_e[dest, source] = del_e
        self.time_factor_synapse = self.dt / self.tau_syn

        """Outputs"""
        if self.debug:
            print('BUILDING OUTPUTS')
        # Figure out how many outputs there actually are, since an output has as many elements as its input population
        outputs = []
        index = 0
        for out in range(len(network.outputs)):
            source_pop = network.outputs[out]['source']
            num_source_neurons = network.populations[source_pop]['number']
            outputs.append([])
            for num in range(num_source_neurons):
                outputs[out].append(index)
                index += 1
        self.num_outputs = index

        self.output_voltage_connectivity = torch.from_numpy(np.zeros([self.num_outputs, self.num_neurons]))  # initialize connectivity matrix
        self.output_spike_connectivity = self.output_voltage_connectivity.clone()
        for out in range(len(network.outputs)):  # iterate over the connections in the network
            wt = network.outputs[out]['weight']  # get the weight
            source_pop = network.outputs[out]['source']  # get the source
            for i in range(len(pops_and_nrns[source_pop])):
                if network.outputs[out]['spiking']:
                    self.output_spike_connectivity[outputs[out][i]][pops_and_nrns[source_pop][i]] = wt  # set the weight in the correct source and destination
                else:
                    self.output_voltage_connectivity[outputs[out][i]][pops_and_nrns[source_pop][i]] = wt  # set the weight in the correct source and destination

        """Debug Prints"""
        if self.debug:
            print('Input Connectivity:')
            print(self.input_connectivity)
            print('g_max_non:')
            print(self.g_max_non)
            print('GmaxSpike:')
            print(self.g_max_spike)
            print('del_e:')
            print(self.del_e)
            print('Output Voltage Connectivity')
            print(self.output_voltage_connectivity)
            print('Output Spike Connectivity:')
            print(self.output_spike_connectivity)
            print('u:')
            print(self.u)
            print('u_last:')
            print(self.u_last)
            print('theta_0:')
            print(self.theta_0)
            print('ThetaLast:')
            print(self.theta_last)
            print('Theta')
            print(self.theta)
            print('\nDONE BUILDING')

        """Move the tensors to the appropriate device"""
        if device == 'cuda':
            if torch.cuda.is_available():
                if self.debug:
                    print("CUDA Device found, using GPU")
                self.device = 'cuda'
            else:
                if self.debug:
                    print('CUDA Device not found, using CPU')
                self.device = 'cpu'
        else:
            self.device = 'cpu'

        self.u_last = self.u_last.to(self.device)
        self.u = self.u.to(self.device)
        self.theta = self.theta.to(self.device)
        self.theta_last = self.theta_last.to(self.device)
        self.input_connectivity = self.input_connectivity.to(self.device)
        self.g_max_non = self.g_max_non.to(self.device)
        self.g_max_spike = self.g_max_spike.to(self.device)
        self.time_factor_synapse = self.time_factor_synapse.to(self.device)
        self.g_spike = self.g_spike.to(self.device)
        self.del_e = self.del_e.to(self.device)
        self.time_factor_membrane = self.time_factor_membrane.to(self.device)
        self.g_m = self.g_m.to(self.device)
        self.i_b = self.i_b.to(self.device)
        self.theta_0 = self.theta_0.to(self.device)
        self.time_factor_threshold = self.time_factor_threshold.to(self.device)
        self.m = self.m.to(self.device)
        self.spikes = self.spikes.to(self.device)
        self.zero = torch.tensor([0],device=self.device)
        self.output_spike_connectivity = self.output_spike_connectivity.to(self.device)
        self.output_voltage_connectivity = self.output_voltage_connectivity.to(self.device)
        self.out = torch.from_numpy(np.zeros([self.num_outputs, self.num_neurons])).to(self.device)
        self.i_app = torch.from_numpy(np.zeros(self.num_neurons)).to(self.device)
        self.g_non = torch.from_numpy(np.zeros([self.num_neurons, self.num_neurons])).to(self.device)
        self.g_syn = torch.from_numpy(np.zeros([self.num_neurons, self.num_neurons])).to(self.device)
        self.i_syn = torch.from_numpy(np.zeros(self.num_neurons)).to(self.device)

    def forward(self, inputs) -> Any:
        self.u_last = self.u.clone()
        self.theta_last = self.theta.clone()
        self.i_app = torch.matmul(self.input_connectivity, inputs)  # Apply external current sources to their destinations
        self.g_non = torch.maximum(self.zero, torch.minimum(self.g_max_non * self.u_last / self.R, self.g_max_non))
        self.g_spike = self.g_spike * (1 - self.time_factor_synapse)
        self.g_syn = self.g_non + self.g_spike
        self.i_syn = torch.sum(self.g_syn * self.del_e, dim=1) - self.u_last * torch.sum(self.g_syn, dim=1)
        self.u = self.u_last + self.time_factor_membrane * (
                -self.g_m * self.u_last + self.i_b + self.i_syn + self.i_app)  # Update membrane potential
        self.theta = self.theta_last + self.time_factor_threshold * (
                -self.theta_last + self.theta_0 + self.m * self.u_last)  # Update the firing thresholds
        self.spikes = torch.sign(torch.minimum(self.zero, self.theta - self.u))  # Compute which neurons have spiked
        self.g_spike = torch.maximum(self.g_spike,
                                     (-self.spikes) * self.g_max_spike)  # Update the conductance of connections which spiked
        self.u = self.u * (self.spikes + 1)  # Reset the membrane voltages of neurons which spiked
        self.out = torch.matmul(self.output_voltage_connectivity, self.u) + torch.matmul(self.output_spike_connectivity, -self.spikes)
        # return out.cpu().numpy()
        return self.out

"""
########################################################################################################################
PYTORCH SPARSE
"""
# TODO: Redo with inheritance
class SNS_Torch_Large(Backend):
    # TODO: Add polynomial mapping
    # TODO: Add synaptic transmission delay
    def __init__(self, network: Network,dtype, **kwargs):
        super().__init__(network, **kwargs)

        #Neurons
        if self.debug:
            print('BUILDING NEURONS')
        # Initialize the vectors
        self.u = torch.from_numpy(np.zeros(self.num_neurons)).to(dtype)
        self.u_last = torch.from_numpy(np.zeros(self.num_neurons)).to(dtype)
        self.spikes = torch.from_numpy(np.zeros(self.num_neurons)).to(dtype)
        c_m = torch.from_numpy(np.zeros(self.num_neurons)).to(dtype)
        self.g_m = torch.from_numpy(np.zeros(self.num_neurons)).to(dtype)
        self.i_b = torch.from_numpy(np.zeros(self.num_neurons)).to(dtype)
        self.theta_0 = torch.from_numpy(np.zeros(self.num_neurons)).to(dtype)
        self.theta = torch.from_numpy(np.zeros(self.num_neurons)).to(dtype)
        self.theta_last = torch.from_numpy(np.zeros(self.num_neurons)).to(dtype)
        self.m = torch.from_numpy(np.zeros(self.num_neurons)).to(dtype)
        tau_theta = torch.from_numpy(np.zeros(self.num_neurons)).to(dtype)

        # iterate over the populations in the network
        pops_and_nrns = []
        index = 0
        for pop in range(len(network.populations)):
            num_neurons = network.populations[pop]['number']  # find the number of neurons in the population
            pops_and_nrns.append([])
            u_last = 0.0
            for num in range(num_neurons):  # for each neuron, copy the parameters over
                c_m[index] = network.populations[pop]['type'].params['membrane_capacitance']
                self.g_m[index] = network.populations[pop]['type'].params['membrane_conductance']
                self.i_b[index] = network.populations[pop]['type'].params['bias']
                self.u_last[index] = u_last
                if isinstance(network.populations[pop]['type'], SpikingNeuron):  # if the neuron is spiking, copy more
                    self.theta_0[index] = network.populations[pop]['type'].params['threshold_initial_value']
                    u_last += network.populations[pop]['type'].params['threshold_initial_value'] / num_neurons
                    self.m[index] = network.populations[pop]['type'].params['threshold_proportionality_constant']
                    tau_theta[index] = network.populations[pop]['type'].params['threshold_time_constant']
                else:  # otherwise, set to the special values for NonSpiking
                    self.theta_0[index] = sys.float_info.max
                    self.m[index] = 0
                    tau_theta[index] = 1
                    u_last += self.R / num_neurons
                pops_and_nrns[pop].append(index)
                index += 1
        self.u = self.u_last.clone().to(dtype)
        # set the derived vectors
        self.time_factor_membrane = (self.dt / c_m).to(dtype)
        self.time_factor_threshold = (self.dt / tau_theta).to(dtype)
        self.theta = self.theta_0.clone().to(dtype)
        self.theta_last = self.theta_0.clone().to(dtype)

        #Inputs
        if self.debug:
            print('BUILDING INPUTS')
        rows = []
        cols = []
        vals = []
        for conn in network.inputConns:  # iterate over the connections in the network
            wt = conn['weight']  # get the weight
            source = conn['source']  # get the source
            dest_pop = conn['destination']  # get the destination
            for dest in pops_and_nrns[dest_pop]:
                rows.append(dest)
                cols.append(source)
                vals.append(wt)

        self.input_connectivity = torch.sparse_coo_tensor([rows, cols], vals, (self.num_neurons, self.num_inputs)).to(dtype)

        #Synapses
        if self.debug:
            print('BUILDING SYNAPSES')
        # initialize the matrices
        self.tauSyn = (torch.from_numpy(np.zeros([self.num_neurons, self.num_neurons])) + 1).to(dtype)

        # iterate over the connections in the network
        non_rows = []
        non_cols = []
        non_vals = []
        spike_rows = []
        spike_cols = []
        spike_vals = []
        spike_cond = []
        del_e_rows = []
        del_e_cols = []
        del_e_vals = []
        ones = []
        for syn in range(len(network.connections)):
            source_pop = network.connections[syn]['source']
            dest_pop = network.connections[syn]['destination']
            g_max = network.connections[syn]['type'].params['max_conductance']
            del_e = network.connections[syn]['type'].params['relative_reversal_potential']

            if isinstance(network.connections[syn]['type'], SpikingSynapse):
                tau_s = network.connections[syn]['type'].params['synapticTimeConstant']
                for source in pops_and_nrns[source_pop]:
                    for dest in pops_and_nrns[dest_pop]:
                        self.tauSyn[dest,source] = tau_s
                        spike_rows.append(dest)
                        spike_cols.append(source)
                        spike_vals.append(g_max / len(pops_and_nrns[source_pop]))
                        del_e_rows.append(dest)
                        del_e_cols.append(source)
                        del_e_vals.append(del_e)
                        spike_cond.append(0.0)
                        ones.append(1.0)
            else:
                for source in pops_and_nrns[source_pop]:
                    for dest in pops_and_nrns[dest_pop]:
                        # self.g_max_non[dest,source] = g_max / len(pops_and_nrns[source_pop])
                        # self.del_e[dest,source] = del_e
                        non_rows.append(dest)
                        non_cols.append(source)
                        non_vals.append(g_max / len(pops_and_nrns[source_pop]))
                        del_e_rows.append(dest)
                        del_e_cols.append(source)
                        del_e_vals.append(del_e)
        self.g_max_non = torch.sparse_coo_tensor([non_rows, non_cols], non_vals, (self.num_neurons, self.num_neurons)).to(dtype)
        self.g_max_spike = torch.sparse_coo_tensor([spike_rows, spike_cols], spike_vals, (self.num_neurons, self.num_neurons)).to(dtype)
        self.del_e = torch.sparse_coo_tensor([del_e_rows, del_e_cols], del_e_vals, (self.num_neurons, self.num_neurons)).to(dtype)
        self.g_spike = torch.sparse_coo_tensor([spike_rows, spike_cols], spike_cond, size=(self.num_neurons, self.num_neurons)).to(dtype)
        self.ones = torch.sparse_coo_tensor([spike_rows,spike_cols], ones, size=(self.num_neurons, self.num_neurons)).to(dtype)
        self.time_factor_synapse = (self.dt / self.tauSyn).to(dtype)

        #Outputs
        if self.debug:
            print('BUILDING OUTPUTS')
        # Figure out how many outputs there actually are, since an output has as many elements as its input population
        outputs = []
        index = 0
        for out in range(len(network.outputs)):
            source_pop = network.outputs[out]['source']
            num_source_neurons = network.populations[source_pop]['number']
            outputs.append([])
            for num in range(num_source_neurons):
                outputs[out].append(index)
                index += 1
        self.num_outputs = index

        # self.output_voltage_connectivity = torch.from_numpy(np.zeros([self.num_outputs, self.shape]))  # initialize connectivity matrix
        # self.output_spike_connectivity = self.output_voltage_connectivity.clone()
        volt_rows = []
        volt_cols = []
        volt_vals = []
        spike_rows = []
        spike_cols = []
        spike_vals = []
        for out in range(len(network.outputs)):  # iterate over the connections in the network
            wt = network.outputs[out]['weight']  # get the weight
            source_pop = network.outputs[out]['source']  # get the source
            for i in range(len(pops_and_nrns[source_pop])):
                if network.outputs[out]['spiking']:
                    # self.output_spike_connectivity[outputs[out][i]][pops_and_nrns[source_pop][i]] = wt  # set the weight in the correct source and destination
                    spike_rows.append(outputs[out][i])
                    spike_cols.append(pops_and_nrns[source_pop][i])
                    spike_vals.append(wt)
                else:
                    # self.output_voltage_connectivity[outputs[out][i]][pops_and_nrns[source_pop][i]] = wt  # set the weight in the correct source and destination
                    volt_rows.append(outputs[out][i])
                    volt_cols.append(pops_and_nrns[source_pop][i])
                    volt_vals.append(wt)
        self.output_voltage_connectivity = torch.sparse_coo_tensor([volt_rows, volt_cols], volt_vals, (self.num_outputs, self.num_neurons)).to(dtype)
        self.output_spike_connectivity = torch.sparse_coo_tensor([spike_rows, spike_cols], spike_vals, (self.num_outputs, self.num_neurons)).to(dtype)

        #DEBUG
        if self.debug:
            print('Input Connectivity:')
            print(self.input_connectivity)
            print('g_max_non:')
            print(self.g_max_non)
            print('GmaxSpike:')
            print(self.g_max_spike)
            print('del_e:')
            print(self.del_e)
            print('Output Voltage Connectivity')
            print(self.output_voltage_connectivity)
            print('Output Spike Connectivity:')
            print(self.output_spike_connectivity)
            print('u:')
            print(self.u)
            print('u_last:')
            print(self.u_last)
            print('theta_0:')
            print(self.theta_0)
            print('ThetaLast:')
            print(self.theta_last)
            print('Theta')
            print(self.theta)
            print('\nDONE BUILDING')

        #Move the tensors
        if torch.cuda.is_available():
            if self.debug:
                print("CUDA Device found, using GPU")
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.input_connectivity = self.input_connectivity.to(dtype)
        self.output_spike_connectivity = self.output_spike_connectivity.to(dtype)
        self.output_voltage_connectivity = self.output_voltage_connectivity.to(dtype)
        self.zero = torch.tensor([0.0],device='cpu',dtype=dtype)

    def forward(self, inputs) -> Any:
        self.u_last = self.u.clone()
        self.theta_last = self.theta.clone()

        [self.input_connectivity, inputs] = send_vars([self.input_connectivity, inputs], self.device)
        i_app = torch.matmul(self.input_connectivity, inputs)  # Apply external current sources to their destinations
        [self.input_connectivity, inputs, i_app] = send_vars([self.input_connectivity, inputs, i_app], 'cpu')

        [self.zero, self.g_max_non, self.u_last] = send_vars([self.zero, self.g_max_non, self.u_last], self.device)
        g_non = torch.maximum(self.zero, torch.minimum(self.g_max_non.to_dense() * (self.u_last / self.R), self.g_max_non.to_dense())).to_sparse()    # Sparse version unsupported
        [self.zero, self.g_max_non, self.u_last, g_non] = send_vars([self.zero, self.g_max_non, self.u_last, g_non], 'cpu')

        [self.g_spike, self.ones, self.time_factor_synapse] = send_vars([self.g_spike, self.ones, self.time_factor_synapse], self.device)
        self.g_spike = self.g_spike * (self.ones - self.time_factor_synapse.to_sparse())    # Sparse version unsupported
        [self.ones, self.time_factor_synapse] = send_vars([self.ones, self.time_factor_synapse], 'cpu')

        [g_non] = send_vars([g_non], self.device)
        g_syn = g_non + self.g_spike
        [g_non, self.g_spike] = send_vars([g_non, self.g_spike], 'cpu')

        [self.del_e, self.u_last] = send_vars([self.del_e, self.u_last], self.device)
        Isyn = (torch.sum(g_syn.to_dense() * self.del_e.to_dense(), dim=1)).to_sparse() - (self.u_last * torch.sum(g_syn.to_dense(), dim=1)).to_sparse()    # Sparse version unsupported
        [g_syn, self.del_e] = send_vars([g_syn, self.del_e], 'cpu')

        [self.u, self.time_factor_membrane, self.g_m, self.i_b, i_app] = send_vars([self.u, self.time_factor_membrane, self.g_m, self.i_b, i_app], self.device)
        self.u = self.u_last + self.time_factor_membrane * (
                -self.g_m * self.u_last + self.i_b + Isyn + i_app)  # Update membrane potential
        [self.u, self.time_factor_membrane, self.g_m, self.i_b, i_app, Isyn] = send_vars(
            [self.u, self.time_factor_membrane, self.g_m, self.i_b, i_app, Isyn], 'cpu')

        [self.theta, self.theta_last, self.time_factor_threshold, self.theta_0, self.m] = send_vars([self.theta, self.theta_last, self.time_factor_threshold, self.theta_0, self.m], self.device)
        self.theta = self.theta_last + self.time_factor_threshold * (
                -self.theta_last + self.theta_0 + self.m * self.u_last)  # Update the firing thresholds
        [self.theta_last, self.time_factor_threshold, self.theta_0, self.m, self.u_last] = send_vars(
            [self.theta_last, self.time_factor_threshold, self.theta_0, self.m, self.u_last], 'cpu')

        [self.spikes, self.zero, self.u] = send_vars([self.spikes, self.zero, self.u], self.device)
        self.spikes = torch.sign(torch.minimum(self.zero, self.theta - self.u))  # Compute which neurons have spiked
        [self.zero, self.theta, self.u] = send_vars([self.zero, self.theta, self.u], 'cpu')

        [self.g_spike, self.g_max_spike] = send_vars([self.g_spike, self.g_max_spike], self.device)
        self.g_spike = torch.maximum(self.g_spike.to_dense(), (-self.spikes) * self.g_max_spike.to_dense()).to_sparse()  # Update the conductance of connections which spiked, sparse version unsupported
        [self.g_spike, self.g_max_spike] = send_vars([self.g_spike, self.g_max_spike], 'cpu')

        [self.u] = send_vars([self.u], self.device)
        self.u = self.u * (self.spikes + 1)  # Reset the membrane voltages of neurons which spiked

        [self.output_voltage_connectivity, self.output_spike_connectivity] = send_vars([self.output_voltage_connectivity, self.output_spike_connectivity], self.device)
        out = torch.matmul(self.output_voltage_connectivity, self.u) + torch.matmul(self.output_spike_connectivity, -self.spikes)
        [out, self.output_voltage_connectivity, self.u, self.output_spike_connectivity, self.spikes] = send_vars(
            [out, self.output_voltage_connectivity, self.u, self.output_spike_connectivity, self.spikes], 'cpu')

        return out
