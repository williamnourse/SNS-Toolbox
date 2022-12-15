from sns_toolbox.backends import SNS_Numpy, SNS_Torch, SNS_Sparse, SNS_Iterative
from sns_toolbox.neurons import SpikingNeuron, NonSpikingNeuronWithGatedChannels

import numpy as np
import torch
import sys
import warnings

def __compile_numpy__(network, dt=0.01, debug=False) -> SNS_Numpy:
    if debug:
        print('-------------------------------------------------------------------------------------------------------')
        print('COMPILING NETWORK USING NUMPY:')
        print('-------------------------------------------------------------------------------------------------------')

    """
    --------------------------------------------------------------------------------------------------------------------
    Get net parameters
    --------------------------------------------------------------------------------------------------------------------
    """
    if debug:
        print('-----------------------------')
        print('Getting network parameters...')
        print('-----------------------------')
    spiking = network.params['spiking']
    delay = network.params['delay']
    electrical = network.params['electrical']
    electrical_rectified = network.params['electricalRectified']
    gated = network.params['gated']
    num_channels = network.params['numChannels']
    name = network.params['name']
    num_populations = network.get_num_populations()
    num_neurons = network.get_num_neurons()
    num_connections = network.get_num_connections()
    num_inputs = network.get_num_inputs()
    num_outputs = network.get_num_outputs()
    # R = network.params['R']
    if debug:
        print('Spiking:')
        print(spiking)
        print('Spiking Propagation Delay:')
        print(delay)
        print('Electrical Synapses:')
        print(electrical)
        print('Rectified Electrical Synapses:')
        print(electrical_rectified)
        print('Number of Populations:')
        print(num_populations)
        print('Number of Neurons:')
        print(num_neurons)
        print('Number of Connections')
        print(num_connections)
        print('Number of Inputs:')
        print(num_inputs)
        print('Number of Outputs:')
        print(num_outputs)
        # print('Network Voltage Range (mV):')
        # print(R)

    """
    --------------------------------------------------------------------------------------------------------------------
    Initialize vectors and matrices
    --------------------------------------------------------------------------------------------------------------------
    """
    if debug:
        print('---------------------------------')
        print('Initializing vectors and matrices')
        print('---------------------------------')
    V = np.zeros(num_neurons)
    V_last = np.zeros(num_neurons)
    V_0 = np.zeros(num_neurons)
    V_rest = np.zeros(num_neurons)
    c_m = np.zeros(num_neurons)
    g_m = np.zeros(num_neurons)
    i_b = np.zeros(num_neurons)
    if spiking:
        spikes = np.zeros(num_neurons)
        theta_0 = np.zeros(num_neurons)
        theta = np.zeros(num_neurons)
        theta_last = np.zeros(num_neurons)
        m = np.zeros(num_neurons)
        tau_theta = np.zeros(num_neurons)

    g_max_non = np.zeros([num_neurons, num_neurons])
    del_e = np.zeros([num_neurons, num_neurons])
    e_lo = np.zeros([num_neurons, num_neurons])
    e_hi = np.ones([num_neurons, num_neurons])
    if spiking:
        g_max_spike = np.zeros([num_neurons, num_neurons])
        g_spike = np.zeros([num_neurons, num_neurons])
        tau_syn = np.zeros([num_neurons, num_neurons]) + 1
        if delay:
            spike_delays = np.zeros([num_neurons, num_neurons])
            spike_rows = []
            spike_cols = []
            buffer_steps = []
            buffer_nrns = []
            delayed_spikes = np.zeros([num_neurons, num_neurons])
    if electrical:
        g_electrical = np.zeros([num_neurons, num_neurons])
    if electrical_rectified:
        g_rectified = np.zeros([num_neurons, num_neurons])
    if gated:
        # Channel params
        g_ion = np.zeros([num_channels, num_neurons])
        e_ion = np.zeros([num_channels, num_neurons])
        # A gate params
        pow_a = np.zeros([num_channels, num_neurons])
        slope_a = np.zeros([num_channels, num_neurons])
        k_a = np.zeros([num_channels, num_neurons]) + 1
        e_a = np.zeros([num_channels, num_neurons])
        # B gate params
        pow_b = np.zeros([num_channels, num_neurons])
        slope_b = np.zeros([num_channels, num_neurons])
        k_b = np.zeros([num_channels, num_neurons]) + 1
        e_b = np.zeros([num_channels, num_neurons])
        tau_max_b = np.zeros([num_channels, num_neurons]) + 1
        # C gate params
        pow_c = np.zeros([num_channels, num_neurons])
        slope_c = np.zeros([num_channels, num_neurons])
        k_c = np.zeros([num_channels, num_neurons]) + 1
        e_c = np.zeros([num_channels, num_neurons])
        tau_max_c = np.zeros([num_channels, num_neurons]) + 1

        b_gate = np.zeros([num_channels, num_neurons])
        b_gate_last = np.zeros([num_channels, num_neurons])
        b_gate_0 = np.zeros([num_channels, num_neurons])
        c_gate = np.zeros([num_channels, num_neurons])
        c_gate_last = np.zeros([num_channels, num_neurons])
        c_gate_0 = np.zeros([num_channels, num_neurons])

    pops_and_nrns = []
    index = 0
    for pop in range(len(network.populations)):
        num_neurons_in_pop = network.populations[pop]['number']  # find the number of neurons in the population
        pops_and_nrns.append([])
        for num in range(num_neurons_in_pop):
            pops_and_nrns[pop].append(index)
            index += 1

    """
    --------------------------------------------------------------------------------------------------------------------
    Set Neurons
    --------------------------------------------------------------------------------------------------------------------
    """
    if debug:
        print('---------------')
        print('Setting neurons')
        print('---------------')
    index = 0
    for pop in range(len(network.populations)):
        num_neurons_in_pop = network.populations[pop]['number']  # find the number of neurons in the population
        initial_value = network.populations[pop]['initial_value']
        for num in range(num_neurons_in_pop):  # for each neuron, copy the parameters over
            c_m[index] = network.populations[pop]['type'].params['membrane_capacitance']
            g_m[index] = network.populations[pop]['type'].params['membrane_conductance']
            i_b[index] = network.populations[pop]['type'].params['bias']
            V_rest[index] = network.populations[pop]['type'].params['resting_potential']
            if hasattr(initial_value, '__iter__'):
                V_last[index] = initial_value[num]
            elif initial_value is None:
                V_last[index] = V_rest[index]
            else:
                V_last[index] = initial_value
            if spiking:
                if isinstance(network.populations[pop]['type'],
                              SpikingNeuron):  # if the neuron is spiking, copy more
                    theta_0[index] = network.populations[pop]['type'].params['threshold_initial_value']
                    m[index] = network.populations[pop]['type'].params['threshold_proportionality_constant']
                    tau_theta[index] = network.populations[pop]['type'].params['threshold_time_constant']
                else:  # otherwise, set to the special values for NonSpiking
                    theta_0[index] = sys.float_info.max
                    m[index] = 0
                    tau_theta[index] = 1
            if gated:
                if isinstance(network.populations[pop]['type'], NonSpikingNeuronWithGatedChannels):
                    # Channel params
                    g_ion[:, index] = network.populations[pop]['type'].params['Gion']
                    e_ion[:, index] = network.populations[pop]['type'].params['Eion']
                    # A gate params
                    pow_a[:, index] = network.populations[pop]['type'].params['paramsA']['pow']
                    slope_a[:, index] = network.populations[pop]['type'].params['paramsA']['slope']
                    k_a[:, index] = network.populations[pop]['type'].params['paramsA']['k']
                    e_a[:, index] = network.populations[pop]['type'].params['paramsA']['reversal']
                    # B gate params
                    pow_b[:, index] = network.populations[pop]['type'].params['paramsB']['pow']
                    slope_b[:, index] = network.populations[pop]['type'].params['paramsB']['slope']
                    k_b[:, index] = network.populations[pop]['type'].params['paramsB']['k']
                    e_b[:, index] = network.populations[pop]['type'].params['paramsB']['reversal']
                    tau_max_b[:, index] = network.populations[pop]['type'].params['paramsB']['TauMax']
                    # C gate params
                    pow_c[:, index] = network.populations[pop]['type'].params['paramsC']['pow']
                    slope_c[:, index] = network.populations[pop]['type'].params['paramsC']['slope']
                    k_c[:, index] = network.populations[pop]['type'].params['paramsC']['k']
                    e_c[:, index] = network.populations[pop]['type'].params['paramsC']['reversal']
                    tau_max_c[:, index] = network.populations[pop]['type'].params['paramsC']['TauMax']

                    b_gate_last[:, index] = 1 / (1 + k_b[:, index] * np.exp(
                        slope_b[:, index] * (V_last[index] - e_b[:, index])))
                    c_gate_last[:, index] = 1 / (1 + k_c[:, index] * np.exp(
                        slope_c[:, index] * (V_last[index] - e_c[:, index])))
            index += 1
    V = np.copy(V_last)
    V_0 = np.copy(V_last)
    if spiking:
        theta = np.copy(theta_0)
        theta_last = np.copy(theta_0)
    if gated:
        b_gate = np.copy(b_gate_last)
        b_gate_0 = np.copy(b_gate_last)
        c_gate = np.copy(c_gate_last)
        c_gate_0 = np.copy(c_gate_last)

    """
    --------------------------------------------------------------------------------------------------------------------
    Set Inputs
    --------------------------------------------------------------------------------------------------------------------
    """
    if debug:
        print('--------------')
        print('Setting Inputs')
        print('--------------')
    input_connectivity = np.zeros(
        [num_neurons, network.get_num_inputs_actual()])  # initialize connectivity matrix
    index = 0
    for inp in range(network.get_num_inputs()):  # iterate over the connections in the network
        size = network.inputs[inp]['size']
        dest_pop = network.inputs[inp]['destination']  # get the destination
        if size == 1:
            for dest in pops_and_nrns[dest_pop]:
                input_connectivity[dest][inp] = 1.0  # set the weight in the correct source and destination
            index += 1
        else:
            for dest in pops_and_nrns[dest_pop]:
                input_connectivity[dest][index] = 1.0
                index += 1

    """
    --------------------------------------------------------------------------------------------------------------------
    Set Connections
    --------------------------------------------------------------------------------------------------------------------
    """
    if debug:
        print('-------------------')
        print('Setting connections')
        print('-------------------')
    for syn in range(len(network.connections)):
        source_pop = network.connections[syn]['source']
        dest_pop = network.connections[syn]['destination']
        g_max = network.connections[syn]['params']['max_conductance']
        del_e_val = None
        e_lo_val = None
        e_hi_val = None
        if network.connections[syn]['params']['electrical'] is False:  # electrical connection
            del_e_val = network.connections[syn]['params']['reversal_potential']

        if network.connections[syn]['params']['matrix']:  # pattern and matrix connections
            pop_size_source = len(pops_and_nrns[source_pop])
            pop_size_dest = len(pops_and_nrns[dest_pop])
            source_index = pops_and_nrns[source_pop][0]
            dest_index = pops_and_nrns[dest_pop][0]
            if network.connections[syn]['params']['spiking']:
                tau_s = network.connections[syn]['params']['synapticTimeConstant']
                g_max_spike[dest_index:dest_index + pop_size_dest, source_index:source_index + pop_size_source] = g_max
                del_e[dest_index:dest_index + pop_size_dest, source_index:source_index + pop_size_source] = del_e_val
                tau_syn[dest_index:dest_index + pop_size_dest, source_index:source_index + pop_size_source] = tau_s
                if delay:
                    delay_val = network.connections[syn]['params']['synapticTransmissionDelay']
                    spike_delays[dest_index:dest_index + pop_size_dest, source_index:source_index + pop_size_source] = delay_val

                    for source in pops_and_nrns[source_pop]:
                        for dest in pops_and_nrns[dest_pop]:
                            buffer_nrns.append(source)
                            buffer_steps.append(delay)
                            spike_rows.append(dest)
                            spike_cols.append(source)
            else:
                e_lo_val = network.connections[syn]['params']['e_lo']
                e_hi_val = network.connections[syn]['params']['e_hi']
                g_max_non[dest_index:dest_index + pop_size_dest, source_index:source_index + pop_size_source] = g_max
                del_e[dest_index:dest_index + pop_size_dest, source_index:source_index + pop_size_source] = del_e_val
                e_lo[dest_index:dest_index + pop_size_dest, source_index:source_index + pop_size_source] = e_lo_val
                e_hi[dest_index:dest_index + pop_size_dest, source_index:source_index + pop_size_source] = e_hi_val
        elif network.connections[syn]['params']['electrical']:  # electrical connection
            for source in pops_and_nrns[source_pop]:
                for dest in pops_and_nrns[dest_pop]:
                    if network.connections[syn]['params']['rectified']:  # rectified
                        g_rectified[dest][source] = g_max / len(pops_and_nrns[source_pop])
                    else:
                        g_electrical[dest][source] = g_max / len(pops_and_nrns[source_pop])
                        g_electrical[source][dest] = g_max / len(pops_and_nrns[source_pop])
        else:  # chemical connection
            if network.connections[syn]['params']['spiking']:  # spiking chemical synapse
                tau_s = network.connections[syn]['params']['synapticTimeConstant']
                if delay:
                    delay_val = network.connections[syn]['params']['synapticTransmissionDelay']
                for source in pops_and_nrns[source_pop]:
                    for dest in pops_and_nrns[dest_pop]:
                        g_max_spike[dest][source] = g_max / len(pops_and_nrns[source_pop])
                        del_e[dest][source] = del_e_val
                        tau_syn[dest][source] = tau_s
                        if delay:
                            spike_delays[dest][source] = delay_val
                            buffer_nrns.append(source)
                            buffer_steps.append(delay_val)
                            spike_rows.append(dest)
                            spike_cols.append(source)
            else:  # nonspiking chemical synapse
                e_lo_val = network.connections[syn]['params']['e_lo']
                e_hi_val = network.connections[syn]['params']['e_hi']
                for source in pops_and_nrns[source_pop]:
                    for dest in pops_and_nrns[dest_pop]:
                        g_max_non[dest][source] = g_max / len(pops_and_nrns[source_pop])
                        del_e[dest][source] = del_e_val
                        e_lo[dest][source] = e_lo_val
                        e_hi[dest][source] = e_hi_val

    """
    --------------------------------------------------------------------------------------------------------------------
    Calculate Time Factors
    --------------------------------------------------------------------------------------------------------------------
    """
    if debug:
        print('------------------------')
        print('Calculating Time Factors')
        print('------------------------')
    time_factor_membrane = dt / (c_m / g_m)
    if spiking:
        time_factor_threshold = dt / tau_theta
        time_factor_synapse = dt / tau_syn

    """
    --------------------------------------------------------------------------------------------------------------------
    Initialize Propagation Delay
    --------------------------------------------------------------------------------------------------------------------
    """
    if delay:
        if debug:
            print('------------------------------')
            print('Initializing Propagation Delay')
            print('------------------------------')
        buffer_length = int(np.max(spike_delays) + 1)
        spike_buffer = np.zeros([buffer_length, num_neurons])

    """
    --------------------------------------------------------------------------------------------------------------------
    Set Outputs
    --------------------------------------------------------------------------------------------------------------------
    """
    if debug:
        print('---------------')
        print('Setting Outputs')
        print('---------------')
    output_nodes = []
    index = 0
    for out in range(len(network.outputs)):
        source_pop = network.outputs[out]['source']
        num_source_neurons = network.populations[source_pop]['number']
        output_nodes.append([])
        for num in range(num_source_neurons):
            output_nodes[out].append(index)
            index += 1
    num_outputs = index

    output_voltage_connectivity = np.zeros(
        [num_outputs, num_neurons])  # initialize connectivity matrix
    if spiking:
        output_spike_connectivity = np.copy(output_voltage_connectivity)
    outputs = np.zeros(num_outputs)
    for out in range(len(network.outputs)):  # iterate over the connections in the network
        source_pop = network.outputs[out]['source']  # get the source
        for i in range(len(pops_and_nrns[source_pop])):
            if network.outputs[out]['spiking']:
                output_spike_connectivity[output_nodes[out][i]][
                    pops_and_nrns[source_pop][i]] = 1.0  # set the weight in the correct source and destination
            else:
                output_voltage_connectivity[output_nodes[out][i]][
                    pops_and_nrns[source_pop][i]] = 1.0  # set the weight in the correct source and destination

    """
    --------------------------------------------------------------------------------------------------------------------
    Arrange states and parameters into dictionary
    --------------------------------------------------------------------------------------------------------------------
    """
    if debug:
        print('---------------------')
        print('Writing to Dictionary')
        print('---------------------')
    params = {'dt': dt,
              'name': name,
              'spiking': spiking,
              'delay': delay,
              'elec': electrical,
              'rect': electrical_rectified,
              'gated': gated,
              'numChannels': num_channels,
              'v': V,
              'vLast': V_last,
              'vRest': V_rest,
              'v0': V_0,
              'cM': c_m,
              'gM': g_m,
              'iB': i_b,
              'gMaxNon': g_max_non,
              'delE': del_e,
              'eLo': e_lo,
              'eHi': e_hi,
              'timeFactorMembrane': time_factor_membrane,
              'inputConn': input_connectivity,
              'numPop': num_populations,
              'numNeurons': num_neurons,
              'numConn': num_connections,
              'numInputs': num_inputs,
              'numOutputs': num_outputs,
              # 'r': R,
              'outConnVolt': output_voltage_connectivity}
    if spiking:
        params['spikes'] = spikes
        params['theta0'] = theta_0
        params['theta'] = theta
        params['thetaLast'] = theta_last
        params['m'] = m
        params['tauTheta'] = tau_theta
        params['gMaxSpike'] = g_max_spike
        params['gSpike'] = g_spike
        params['tauSyn'] = tau_syn
        params['timeFactorThreshold'] = time_factor_threshold
        params['timeFactorSynapse'] = time_factor_synapse
        params['outConnSpike'] = output_spike_connectivity
    if delay:
        params['spikeDelays'] = spike_delays
        params['spikeRows'] = spike_rows
        params['spikeCols'] = spike_cols
        params['bufferSteps'] = buffer_steps
        params['bufferNrns'] = buffer_nrns
        params['delayedSpikes'] = delayed_spikes
        params['spikeBuffer'] = spike_buffer
    if electrical:
        params['gElectrical'] = g_electrical
    if electrical_rectified:
        params['gRectified'] = g_rectified
    if gated:
        params['gIon'] = g_ion
        params['eIon'] = e_ion
        params['powA'] = pow_a
        params['slopeA'] = slope_a
        params['kA'] = k_a
        params['eA'] = e_a
        params['powB'] = pow_b
        params['slopeB'] = slope_b
        params['kB'] = k_b
        params['eB'] = e_b
        params['tauMaxB'] = tau_max_b
        params['powC'] = pow_c
        params['slopeC'] = slope_c
        params['kC'] = k_c
        params['eC'] = e_c
        params['tauMaxC'] = tau_max_c
        params['bGate'] = b_gate
        params['bGateLast'] = b_gate_last
        params['bGate0'] = b_gate_0
        params['cGate'] = c_gate
        params['cGateLast'] = c_gate_last
        params['cGate0'] = c_gate_0

    """
    --------------------------------------------------------------------------------------------------------------------
    Passing params to backend object
    --------------------------------------------------------------------------------------------------------------------
    """
    if debug:
        print('-------------------------------------------------')
        print('Passing states and parameters to SNS_Numpy object')
        print('-------------------------------------------------')
    model = SNS_Numpy(params)

    """
    --------------------------------------------------------------------------------------------------------------------
    Final print
    --------------------------------------------------------------------------------------------------------------------
    """
    if debug:
        print('----------------------------')
        print('Final states and parameters:')
        print('----------------------------')
        print('Input Connectivity:')
        print(input_connectivity)
        print('g_max_non:')
        print(g_max_non)
        if spiking:
            print('GmaxSpike:')
            print(g_max_spike)
        print('del_e:')
        print(del_e)
        print('e_lo:')
        print(e_lo)
        print('e_hi:')
        print(e_hi)
        if electrical:
            print('Gelectrical:')
            print(g_electrical)
        if electrical_rectified:
            print('GelectricalRectified:')
            print(g_rectified)
        print('Output Voltage Connectivity')
        print(output_voltage_connectivity)
        if spiking:
            print('Output Spike Connectivity:')
            print(output_spike_connectivity)
        print('v:')
        print(V)
        print('v_last:')
        print(V_last)
        print('v_rest:')
        print(V_rest)
        if spiking:
            print('theta_0:')
            print(theta_0)
            print('ThetaLast:')
            print(theta_last)
            print('Theta')
            print(theta)
        if gated:
            print('Number of Channels:')
            print(num_channels)
            print('Ionic Conductance:')
            print(g_ion)
            print('Ionic Reversal Potentials:')
            print(e_ion)
            print('A Gate Parameters:')
            print('Power:')
            print(pow_a)
            print('Slope:')
            print(slope_a)
            print('K:')
            print(k_a)
            print('Reversal Potential:')
            print(e_a)
            print('B Gate Parameters:')
            print('Power:')
            print(pow_b)
            print('Slope:')
            print(slope_b)
            print('K:')
            print(k_b)
            print('Reversal Potential:')
            print(e_b)
            print('Tau Max:')
            print(tau_max_b)
            print('B:')
            print(b_gate)
            print('B_last:')
            print(b_gate_last)
            print('C Gate Parameters:')
            print('Power:')
            print(pow_c)
            print('Slope:')
            print(slope_c)
            print('K:')
            print(k_c)
            print('Reversal Potential:')
            print(e_c)
            print('Tau Max:')
            print(tau_max_c)
            print('B:')
            print(c_gate)
            print('B_last:')
            print(c_gate_last)

    return model

def __compile_torch__(network, dt=0.01, debug=False, device='cpu') -> SNS_Torch:
    if debug:
        print('-------------------------------------------------------------------------------------------------------')
        print('COMPILING NETWORK USING TORCH:')
        print('-------------------------------------------------------------------------------------------------------')
    """
    --------------------------------------------------------------------------------------------------------------------
    Setting hardware device
    --------------------------------------------------------------------------------------------------------------------
    """
    if debug:
        print('-----------------------')
        print('Setting hardware device')
        print('-----------------------')
    if device != 'cpu':
        if not torch.cuda.is_available():
            warnings.warn('Warning: CUDA device not found, switching to CPU')
            device = 'cpu'
    if debug:
        print('Device:')
        print(device)

    """
    --------------------------------------------------------------------------------------------------------------------
    Get net parameters
    --------------------------------------------------------------------------------------------------------------------
    """
    if debug:
        print('-----------------------------')
        print('Getting network parameters...')
        print('-----------------------------')
    spiking = network.params['spiking']
    delay = network.params['delay']
    electrical = network.params['electrical']
    electrical_rectified = network.params['electricalRectified']
    gated = network.params['gated']
    num_channels = network.params['numChannels']
    name = network.params['name']
    num_populations = network.get_num_populations()
    num_neurons = network.get_num_neurons()
    num_connections = network.get_num_connections()
    num_inputs = network.get_num_inputs()
    num_outputs = network.get_num_outputs()
    # R = network.params['R']
    if debug:
        print('Spiking:')
        print(spiking)
        print('Spiking Propagation Delay:')
        print(delay)
        print('Electrical Synapses:')
        print(electrical)
        print('Rectified Electrical Synapses:')
        print(electrical_rectified)
        print('Number of Populations:')
        print(num_populations)
        print('Number of Neurons:')
        print(num_neurons)
        print('Number of Connections')
        print(num_connections)
        print('Number of Inputs:')
        print(num_inputs)
        print('Number of Outputs:')
        print(num_outputs)
        # print('Network Voltage Range (mV):')
        # print(R)

    """
    --------------------------------------------------------------------------------------------------------------------
    Initialize vectors and matrices
    --------------------------------------------------------------------------------------------------------------------
    """
    if debug:
        print('---------------------------------')
        print('Initializing vectors and matrices')
        print('---------------------------------')
    V = torch.zeros(num_neurons, device=device)
    V_last = torch.zeros(num_neurons, device=device)
    V_rest = torch.zeros(num_neurons, device=device)
    V_0 = torch.zeros(num_neurons, device=device)
    c_m = torch.zeros(num_neurons, device=device)
    g_m = torch.zeros(num_neurons, device=device)
    i_b = torch.zeros(num_neurons, device=device)
    if spiking:
        spikes = torch.zeros(num_neurons, device=device)
        theta_0 = torch.zeros(num_neurons, device=device)
        theta = torch.zeros(num_neurons, device=device)
        theta_last = torch.zeros(num_neurons, device=device)
        m = torch.zeros(num_neurons, device=device)
        tau_theta = torch.zeros(num_neurons, device=device)

    g_max_non = torch.zeros([num_neurons, num_neurons], device=device)
    del_e = torch.zeros([num_neurons, num_neurons], device=device)
    e_lo = torch.zeros([num_neurons, num_neurons], device=device)
    e_hi = torch.ones([num_neurons, num_neurons], device=device)
    if spiking:
        g_max_spike = torch.zeros([num_neurons, num_neurons], device=device)
        g_spike = torch.zeros([num_neurons, num_neurons], device=device)
        tau_syn = torch.ones([num_neurons, num_neurons], device=device)
        if delay:
            spike_delays = torch.zeros([num_neurons, num_neurons], device=device)
            spike_rows = []
            spike_cols = []
            buffer_steps = []
            buffer_nrns = []
            delayed_spikes = torch.zeros([num_neurons, num_neurons], device=device)
    if electrical:
        g_electrical = torch.zeros([num_neurons, num_neurons], device=device)
    if electrical_rectified:
        g_rectified = torch.zeros([num_neurons, num_neurons], device=device)
    if gated:
        # Channel params
        g_ion = torch.zeros([num_channels, num_neurons], device=device)
        e_ion = torch.zeros([num_channels, num_neurons], device=device)
        # A gate params
        pow_a = torch.zeros([num_channels, num_neurons], device=device)
        slope_a = torch.zeros([num_channels, num_neurons], device=device)
        k_a = torch.ones([num_channels, num_neurons], device=device)
        e_a = torch.zeros([num_channels, num_neurons], device=device)
        # B gate params
        pow_b = torch.zeros([num_channels, num_neurons], device=device)
        slope_b = torch.zeros([num_channels, num_neurons], device=device)
        k_b = torch.ones([num_channels, num_neurons], device=device)
        e_b = torch.zeros([num_channels, num_neurons], device=device)
        tau_max_b = torch.ones([num_channels, num_neurons], device=device)
        # C gate params
        pow_c = torch.zeros([num_channels, num_neurons], device=device)
        slope_c = torch.zeros([num_channels, num_neurons], device=device)
        k_c = torch.ones([num_channels, num_neurons], device=device)
        e_c = torch.zeros([num_channels, num_neurons], device=device)
        tau_max_c = torch.ones([num_channels, num_neurons], device=device) + 1

        b_gate = torch.zeros([num_channels, num_neurons], device=device)
        b_gate_last = torch.zeros([num_channels, num_neurons], device=device)
        b_gate_0 = torch.zeros([num_channels, num_neurons], device=device)
        c_gate = torch.zeros([num_channels, num_neurons], device=device)
        c_gate_last = torch.zeros([num_channels, num_neurons], device=device)
        c_gate_0 = torch.zeros([num_channels, num_neurons], device=device)

    pops_and_nrns = []
    index = 0
    for pop in range(len(network.populations)):
        num_neurons_in_pop = network.populations[pop]['number']  # find the number of neurons in the population
        pops_and_nrns.append([])
        for num in range(num_neurons_in_pop):
            pops_and_nrns[pop].append(index)
            index += 1

    """
    --------------------------------------------------------------------------------------------------------------------
    Set Neurons
    --------------------------------------------------------------------------------------------------------------------
    """
    if debug:
        print('---------------')
        print('Setting neurons')
        print('---------------')
    index = 0
    for pop in range(len(network.populations)):
        num_neurons_in_pop = network.populations[pop]['number']  # find the number of neurons in the population
        initial_value = network.populations[pop]['initial_value']
        for num in range(num_neurons_in_pop):  # for each neuron, copy the parameters over
            c_m[index] = network.populations[pop]['type'].params['membrane_capacitance']
            g_m[index] = network.populations[pop]['type'].params['membrane_conductance']
            i_b[index] = network.populations[pop]['type'].params['bias']
            V_rest[index] = network.populations[pop]['type'].params['resting_potential']
            if hasattr(initial_value, '__iter__'):
                V_last[index] = initial_value[num]
            elif initial_value is None:
                V_last[index] = V_rest[index]
            else:
                V_last[index] = initial_value

            if spiking:
                if isinstance(network.populations[pop]['type'],
                              SpikingNeuron):  # if the neuron is spiking, copy more
                    theta_0[index] = network.populations[pop]['type'].params['threshold_initial_value']
                    m[index] = network.populations[pop]['type'].params['threshold_proportionality_constant']
                    tau_theta[index] = network.populations[pop]['type'].params['threshold_time_constant']
                else:  # otherwise, set to the special values for NonSpiking
                    theta_0[index] = torch.finfo(theta_0[index].dtype).max
                    m[index] = 0
                    tau_theta[index] = 1
            if gated:
                if isinstance(network.populations[pop]['type'], NonSpikingNeuronWithGatedChannels):
                    # Channel params
                    g_ion[:, index] = network.populations[pop]['type'].params['Gion']
                    e_ion[:, index] = network.populations[pop]['type'].params['Eion']
                    # A gate params
                    pow_a[:, index] = network.populations[pop]['type'].params['paramsA']['pow']
                    slope_a[:, index] = network.populations[pop]['type'].params['paramsA']['slope']
                    k_a[:, index] = network.populations[pop]['type'].params['paramsA']['k']
                    e_a[:, index] = network.populations[pop]['type'].params['paramsA']['reversal']
                    # B gate params
                    pow_b[:, index] = network.populations[pop]['type'].params['paramsB']['pow']
                    slope_b[:, index] = network.populations[pop]['type'].params['paramsB']['slope']
                    k_b[:, index] = network.populations[pop]['type'].params['paramsB']['k']
                    e_b[:, index] = network.populations[pop]['type'].params['paramsB']['reversal']
                    tau_max_b[:, index] = network.populations[pop]['type'].params['paramsB']['TauMax']
                    # C gate params
                    pow_c[:, index] = network.populations[pop]['type'].params['paramsC']['pow']
                    slope_c[:, index] = network.populations[pop]['type'].params['paramsC']['slope']
                    k_c[:, index] = network.populations[pop]['type'].params['paramsC']['k']
                    e_c[:, index] = network.populations[pop]['type'].params['paramsC']['reversal']
                    tau_max_c[:, index] = network.populations[pop]['type'].params['paramsC']['TauMax']

                    b_gate_last[:, index] = 1 / (1 + k_b[:, index] * torch.exp(
                        slope_b[:, index] * (V_last[index] - e_b[:, index])))
                    c_gate_last[:, index] = 1 / (1 + k_c[:, index] * torch.exp(
                        slope_c[:, index] * (V_last[index] - e_c[:, index])))
            index += 1
    V = V_last.clone()
    if spiking:
        theta = theta_0.clone()
        theta_last = theta_0.clone()
    if gated:
        b_gate = torch.clone(b_gate_last)
        c_gate = torch.clone(c_gate_last)
        b_gate_0 = torch.clone(b_gate_last)
        c_gate_0 = torch.clone(c_gate_last)

    """
    --------------------------------------------------------------------------------------------------------------------
    Set Inputs
    --------------------------------------------------------------------------------------------------------------------
    """
    if debug:
        print('--------------')
        print('Setting Inputs')
        print('--------------')
    input_connectivity = torch.zeros([num_neurons, network.get_num_inputs_actual()],
                                          device=device)  # initialize connectivity matrix
    index = 0
    for inp in range(network.get_num_inputs()):  # iterate over the connections in the network
        size = network.inputs[inp]['size']
        dest_pop = network.inputs[inp]['destination']  # get the destination
        if size == 1:
            for dest in pops_and_nrns[dest_pop]:
                input_connectivity[dest][inp] = 1.0  # set the weight in the correct source and destination
            index += 1
        else:
            for dest in pops_and_nrns[dest_pop]:
                input_connectivity[dest][index] = 1.0
                index += 1

    """
    --------------------------------------------------------------------------------------------------------------------
    Set Connections
    --------------------------------------------------------------------------------------------------------------------
    """
    if debug:
        print('-------------------')
        print('Setting connections')
        print('-------------------')
    for syn in range(len(network.connections)):
        source_pop = network.connections[syn]['source']
        dest_pop = network.connections[syn]['destination']
        g_max_val = network.connections[syn]['params']['max_conductance']
        if network.connections[syn]['params']['electrical'] is False:  # Chemical connection
            del_e_val = network.connections[syn]['params']['reversal_potential']

        if network.connections[syn]['params']['matrix']:  # pattern and connections
            pop_size_source = len(pops_and_nrns[source_pop])
            pop_size_dest = len(pops_and_nrns[dest_pop])
            source_index = pops_and_nrns[source_pop][0]
            dest_index = pops_and_nrns[dest_pop][0]

            if network.connections[syn]['params']['spiking']:
                tau_s = network.connections[syn]['params']['synapticTimeConstant']
                g_max_spike[dest_index:dest_index + pop_size_dest,
                source_index:source_index + pop_size_source] = torch.from_numpy(g_max_val)
                del_e[dest_index:dest_index + pop_size_dest, source_index:source_index + pop_size_source] = torch.from_numpy(
                    del_e_val)
                tau_syn[dest_index:dest_index + pop_size_dest, source_index:source_index + pop_size_source] = torch.from_numpy(
                    tau_s)
                if delay:
                    delay_val = network.connections[syn]['params']['synapticTransmissionDelay']
                    spike_delays[dest_index:dest_index + pop_size_dest,
                    source_index:source_index + pop_size_source] = torch.from_numpy(delay_val)

                    for source in pops_and_nrns[source_pop]:
                        for dest in pops_and_nrns[dest_pop]:
                            buffer_nrns.append(source)
                            buffer_steps.append(delay_val)
                            spike_rows.append(dest)
                            spike_cols.append(source)
            else:
                e_lo_val = network.connections[syn]['params']['e_lo']
                e_hi_val = network.connections[syn]['params']['e_hi']
                g_max_non[dest_index:dest_index + pop_size_dest,
                source_index:source_index + pop_size_source] = torch.from_numpy(g_max_val)
                del_e[dest_index:dest_index + pop_size_dest, source_index:source_index + pop_size_source] = torch.from_numpy(
                    del_e_val)
                e_lo[dest_index:dest_index + pop_size_dest, source_index:source_index + pop_size_source] = torch.from_numpy(
                    e_lo_val)
                e_hi[dest_index:dest_index + pop_size_dest, source_index:source_index + pop_size_source] = torch.from_numpy(
                    e_hi_val)
        elif network.connections[syn]['params']['electrical']:  # electrical connection
            for source in pops_and_nrns[source_pop]:
                for dest in pops_and_nrns[dest_pop]:
                    if network.connections[syn]['params']['rectified']:  # rectified
                        g_rectified[dest][source] = g_max_val / len(pops_and_nrns[source_pop])
                    else:
                        g_electrical[dest][source] = g_max_val / len(pops_and_nrns[source_pop])
                        g_electrical[source][dest] = g_max_val / len(pops_and_nrns[source_pop])
        else:  # chemical connection
            if network.connections[syn]['params']['spiking']:  # spiking chemical synapse
                tau_s = network.connections[syn]['params']['synapticTimeConstant']
                if delay:
                    delay_val = network.connections[syn]['params']['synapticTransmissionDelay']
                for source in pops_and_nrns[source_pop]:
                    for dest in pops_and_nrns[dest_pop]:
                        g_max_spike[dest][source] = g_max_val / len(pops_and_nrns[source_pop])
                        del_e[dest][source] = del_e_val
                        tau_syn[dest][source] = tau_s
                        if delay:
                            spike_delays[dest][source] = delay_val
                            buffer_nrns.append(source)
                            buffer_steps.append(delay_val)
                            spike_rows.append(dest)
                            spike_cols.append(source)
            else:  # nonspiking chemical synapse
                e_lo_val = network.connections[syn]['params']['e_lo']
                e_hi_val = network.connections[syn]['params']['e_hi']
                for source in pops_and_nrns[source_pop]:
                    for dest in pops_and_nrns[dest_pop]:
                        g_max_non[dest][source] = g_max_val / len(pops_and_nrns[source_pop])
                        del_e[dest][source] = del_e_val
                        e_lo[dest][source] = e_lo_val
                        e_hi[dest][source] = e_hi_val

    """
    --------------------------------------------------------------------------------------------------------------------
    Calculate Time Factors
    --------------------------------------------------------------------------------------------------------------------
    """
    if debug:
        print('------------------------')
        print('Calculating Time Factors')
        print('------------------------')
    time_factor_membrane = dt / (c_m / g_m)
    if spiking:
        time_factor_threshold = dt / tau_theta
        time_factor_synapse = dt / tau_syn

    """
    --------------------------------------------------------------------------------------------------------------------
    Initialize Propagation Delay
    --------------------------------------------------------------------------------------------------------------------
    """
    if delay:
        if debug:
            print('------------------------------')
            print('Initializing Propagation Delay')
            print('------------------------------')
        buffer_length = int(torch.max(spike_delays) + 1)
        spike_buffer = torch.zeros([buffer_length, num_neurons], device=device)

    """
    --------------------------------------------------------------------------------------------------------------------
    Set Outputs
    --------------------------------------------------------------------------------------------------------------------
    """
    if debug:
        print('---------------')
        print('Setting Outputs')
        print('---------------')
    output_nodes = []
    index = 0
    for out in range(len(network.outputs)):
        source_pop = network.outputs[out]['source']
        num_source_neurons = network.populations[source_pop]['number']
        output_nodes.append([])
        for num in range(num_source_neurons):
            output_nodes[out].append(index)
            index += 1
    num_outputs = index

    output_voltage_connectivity = torch.zeros(
        [num_outputs, num_neurons], device=device)  # initialize connectivity matrix
    if spiking:
        output_spike_connectivity = torch.clone(output_voltage_connectivity)
    outputs = torch.zeros(num_outputs, device=device)
    for out in range(len(network.outputs)):  # iterate over the connections in the network
        source_pop = network.outputs[out]['source']  # get the source
        for i in range(len(pops_and_nrns[source_pop])):
            if network.outputs[out]['spiking']:
                output_spike_connectivity[output_nodes[out][i]][
                    pops_and_nrns[source_pop][i]] = 1.0  # set the weight in the correct source and destination
                # out_linear[outputs[out][i]] = 1.0
            else:
                output_voltage_connectivity[output_nodes[out][i]][
                    pops_and_nrns[source_pop][i]] = 1.0  # set the weight in the correct source and destination

    """
    --------------------------------------------------------------------------------------------------------------------
    Arrange states and parameters into dictionary
    --------------------------------------------------------------------------------------------------------------------
    """
    if debug:
        print('---------------------')
        print('Writing to Dictionary')
        print('---------------------')
    params = {'dt': dt,
              'name': name,
              'spiking': spiking,
              'delay': delay,
              'elec': electrical,
              'rect': electrical_rectified,
              'gated': gated,
              'numChannels': num_channels,
              'v': V,
              'vLast': V_last,
              'vRest': V_rest,
              'v0': V_0,
              'cM': c_m,
              'gM': g_m,
              'iB': i_b,
              'gMaxNon': g_max_non,
              'delE': del_e,
              'eLo': e_lo,
              'eHi': e_hi,
              'timeFactorMembrane': time_factor_membrane,
              'inputConn': input_connectivity,
              'numPop': num_populations,
              'numNeurons': num_neurons,
              'numConn': num_connections,
              'numInputs': num_inputs,
              'numOutputs': num_outputs,
              # 'r': R,
              'outConnVolt': output_voltage_connectivity}
    if spiking:
        params['spikes'] = spikes
        params['theta0'] = theta_0
        params['theta'] = theta
        params['thetaLast'] = theta_last
        params['m'] = m
        params['tauTheta'] = tau_theta
        params['gMaxSpike'] = g_max_spike
        params['gSpike'] = g_spike
        params['tauSyn'] = tau_syn
        params['timeFactorThreshold'] = time_factor_threshold
        params['timeFactorSynapse'] = time_factor_synapse
        params['outConnSpike'] = output_spike_connectivity
    if delay:
        params['spikeDelays'] = spike_delays
        params['spikeRows'] = spike_rows
        params['spikeCols'] = spike_cols
        params['bufferSteps'] = buffer_steps
        params['bufferNrns'] = buffer_nrns
        params['delayedSpikes'] = delayed_spikes
        params['spikeBuffer'] = spike_buffer
    if electrical:
        params['gElectrical'] = g_electrical
    if electrical_rectified:
        params['gRectified'] = g_rectified
    if gated:
        params['gIon'] = g_ion
        params['eIon'] = e_ion
        params['powA'] = pow_a
        params['slopeA'] = slope_a
        params['kA'] = k_a
        params['eA'] = e_a
        params['powB'] = pow_b
        params['slopeB'] = slope_b
        params['kB'] = k_b
        params['eB'] = e_b
        params['tauMaxB'] = tau_max_b
        params['powC'] = pow_c
        params['slopeC'] = slope_c
        params['kC'] = k_c
        params['eC'] = e_c
        params['tauMaxC'] = tau_max_c
        params['bGate'] = b_gate
        params['bGateLast'] = b_gate_last
        params['bGate0'] = b_gate_0
        params['cGate'] = c_gate
        params['cGateLast'] = c_gate_last
        params['cGate0'] = c_gate_0

    """
    --------------------------------------------------------------------------------------------------------------------
    Passing params to backend object
    --------------------------------------------------------------------------------------------------------------------
    """
    if debug:
        print('-------------------------------------------------')
        print('Passing states and parameters to SNS_Numpy object')
        print('-------------------------------------------------')
    model = SNS_Torch(params)

    """
    --------------------------------------------------------------------------------------------------------------------
    Final print
    --------------------------------------------------------------------------------------------------------------------
    """
    if debug:
        print('----------------------------')
        print('Final states and parameters:')
        print('----------------------------')
        print('Input Connectivity:')
        print(input_connectivity)
        print('g_max_non:')
        print(g_max_non)
        if spiking:
            print('GmaxSpike:')
            print(g_max_spike)
        print('del_e:')
        print(del_e)
        print('e_lo:')
        print(e_lo)
        print('e_hi:')
        print(e_hi)
        if electrical:
            print('Gelectrical:')
            print(g_electrical)
        if electrical_rectified:
            print('GelectricalRectified:')
            print(g_rectified)
        print('Output Voltage Connectivity')
        print(output_voltage_connectivity)
        if spiking:
            print('Output Spike Connectivity:')
            print(output_spike_connectivity)
        print('v:')
        print(V)
        print('v_last:')
        print(V_last)
        print('v_rest:')
        print(V_rest)
        if spiking:
            print('theta_0:')
            print(theta_0)
            print('ThetaLast:')
            print(theta_last)
            print('Theta')
            print(theta)
        if gated:
            print('Number of Channels:')
            print(num_channels)
            print('Ionic Conductance:')
            print(g_ion)
            print('Ionic Reversal Potentials:')
            print(e_ion)
            print('A Gate Parameters:')
            print('Power:')
            print(pow_a)
            print('Slope:')
            print(slope_a)
            print('K:')
            print(k_a)
            print('Reversal Potential:')
            print(e_a)
            print('B Gate Parameters:')
            print('Power:')
            print(pow_b)
            print('Slope:')
            print(slope_b)
            print('K:')
            print(k_b)
            print('Reversal Potential:')
            print(e_b)
            print('Tau Max:')
            print(tau_max_b)
            print('B:')
            print(b_gate)
            print('B_last:')
            print(b_gate_last)
            print('C Gate Parameters:')
            print('Power:')
            print(pow_c)
            print('Slope:')
            print(slope_c)
            print('K:')
            print(k_c)
            print('Reversal Potential:')
            print(e_c)
            print('Tau Max:')
            print(tau_max_c)
            print('B:')
            print(c_gate)
            print('B_last:')
            print(c_gate_last)

    return model

def __compile_sparse__(network, dt=0.01, debug=False, device='cpu') -> SNS_Sparse:
    if debug:
        print('-------------------------------------------------------------------------------------------------------')
        print('COMPILING NETWORK USING TORCH SPARSE:')
        print('-------------------------------------------------------------------------------------------------------')
    """
    --------------------------------------------------------------------------------------------------------------------
    Setting hardware device
    --------------------------------------------------------------------------------------------------------------------
    """
    if debug:
        print('-----------------------')
        print('Setting hardware device')
        print('-----------------------')
    if device != 'cpu':
        if not torch.cuda.is_available():
            warnings.warn('Warning: CUDA device not found, switching to CPU')
            device = 'cpu'
    if debug:
        print('Device:')
        print(device)

    """
    --------------------------------------------------------------------------------------------------------------------
    Get net parameters
    --------------------------------------------------------------------------------------------------------------------
    """
    if debug:
        print('-----------------------------')
        print('Getting network parameters...')
        print('-----------------------------')
    spiking = network.params['spiking']
    delay = network.params['delay']
    electrical = network.params['electrical']
    electrical_rectified = network.params['electricalRectified']
    gated = network.params['gated']
    num_channels = network.params['numChannels']
    name = network.params['name']
    num_populations = network.get_num_populations()
    num_neurons = network.get_num_neurons()
    num_connections = network.get_num_connections()
    num_inputs = network.get_num_inputs()
    num_outputs = network.get_num_outputs()
    # R = network.params['R']
    if debug:
        print('Spiking:')
        print(spiking)
        print('Spiking Propagation Delay:')
        print(delay)
        print('Electrical Synapses:')
        print(electrical)
        print('Rectified Electrical Synapses:')
        print(electrical_rectified)
        print('Number of Populations:')
        print(num_populations)
        print('Number of Neurons:')
        print(num_neurons)
        print('Number of Connections')
        print(num_connections)
        print('Number of Inputs:')
        print(num_inputs)
        print('Number of Outputs:')
        print(num_outputs)
        # print('Network Voltage Range (mV):')
        # print(R)

    """
    --------------------------------------------------------------------------------------------------------------------
    Initialize vectors and matrices
    --------------------------------------------------------------------------------------------------------------------
    """
    if debug:
        print('---------------------------------')
        print('Initializing vectors and matrices')
        print('---------------------------------')
    u = torch.zeros(num_neurons, device=device)
    u_0 = torch.zeros(num_neurons, device=device)
    u_last = torch.zeros(num_neurons, device=device)
    u_rest = torch.zeros(num_neurons, device=device)
    c_m = torch.zeros(num_neurons, device=device)
    g_m = torch.zeros(num_neurons, device=device)
    i_b = torch.sparse_coo_tensor(size=(1, num_neurons), device=device)
    if spiking:
        spikes = torch.sparse_coo_tensor(size=(1, num_neurons), device=device)
        theta_0 = torch.zeros(num_neurons, device=device)
        theta = torch.zeros(num_neurons, device=device)
        theta_last = torch.zeros(num_neurons, device=device)
        m = torch.sparse_coo_tensor(size=(1, num_neurons), device=device)
        tau_theta = torch.zeros(num_neurons, device=device)

    g_max_non = torch.sparse_coo_tensor(size=(num_neurons, num_neurons), device=device)
    del_e = torch.sparse_coo_tensor(size=(num_neurons, num_neurons), device=device)
    e_lo = torch.sparse_coo_tensor(size=(num_neurons, num_neurons), device=device)
    e_hi = torch.ones([num_neurons, num_neurons], device=device)
    if spiking:
        g_max_spike = torch.sparse_coo_tensor(size=(num_neurons, num_neurons), device=device)
        g_spike = torch.sparse_coo_tensor(size=(num_neurons, num_neurons), device=device)
        tau_syn = torch.ones([num_neurons, num_neurons], device=device)
        if delay:
            spike_delays = torch.sparse_coo_tensor(size=(num_neurons, num_neurons), device=device)
            spike_rows = []
            spike_cols = []
            buffer_steps = []
            buffer_nrns = []
            delayed_spikes = torch.sparse_coo_tensor(size=(num_neurons, num_neurons), device=device)
    if electrical:
        g_electrical = torch.sparse_coo_tensor(size=(num_neurons, num_neurons), device=device)
    if electrical_rectified:
        g_rectified = torch.sparse_coo_tensor(size=(num_neurons, num_neurons), device=device)
    if gated:
        # Channel params
        g_ion = torch.sparse_coo_tensor(size=(num_channels, num_neurons), device=device)
        e_ion = torch.sparse_coo_tensor(size=(num_channels, num_neurons), device=device)
        # A gate params
        pow_a = torch.sparse_coo_tensor(size=(num_channels, num_neurons), device=device)
        slope_a = torch.sparse_coo_tensor(size=(num_channels, num_neurons), device=device)
        k_a = torch.ones([num_channels, num_neurons], device=device)
        e_a = torch.sparse_coo_tensor(size=(num_channels, num_neurons), device=device)
        # B gate params
        pow_b = torch.sparse_coo_tensor(size=(num_channels, num_neurons), device=device)
        slope_b = torch.sparse_coo_tensor(size=(num_channels, num_neurons), device=device)
        k_b = torch.ones([num_channels, num_neurons], device=device)
        e_b = torch.sparse_coo_tensor(size=(num_channels, num_neurons), device=device)
        tau_max_b = torch.ones([num_channels, num_neurons], device=device)
        # C gate params
        pow_c = torch.sparse_coo_tensor(size=(num_channels, num_neurons), device=device)
        slope_c = torch.sparse_coo_tensor(size=(num_channels, num_neurons), device=device)
        k_c = torch.ones([num_channels, num_neurons], device=device)
        e_c = torch.sparse_coo_tensor(size=(num_channels, num_neurons), device=device)
        tau_max_c = torch.ones([num_channels, num_neurons], device=device)

        b_gate = torch.sparse_coo_tensor(size=(num_channels, num_neurons), device=device)
        b_gate_last = torch.sparse_coo_tensor(size=(num_channels, num_neurons), device=device)
        b_gate_0 = torch.sparse_coo_tensor(size=(num_channels, num_neurons), device=device)
        c_gate = torch.sparse_coo_tensor(size=(num_channels, num_neurons), device=device)
        c_gate_last = torch.sparse_coo_tensor(size=(num_channels, num_neurons), device=device)
        c_gate_0 = torch.sparse_coo_tensor(size=(num_channels, num_neurons), device=device)

    pops_and_nrns = []
    index = 0
    for pop in range(len(network.populations)):
        num_neurons_in_pop = network.populations[pop]['number']  # find the number of neurons in the population
        pops_and_nrns.append([])
        for num in range(num_neurons_in_pop):
            pops_and_nrns[pop].append(index)
            index += 1

    """
    --------------------------------------------------------------------------------------------------------------------
    Set Neurons
    --------------------------------------------------------------------------------------------------------------------
    """
    if debug:
        print('---------------')
        print('Setting neurons')
        print('---------------')
    index = 0
    for pop in range(len(network.populations)):
        num_neurons_in_pop = network.populations[pop]['number']  # find the number of neurons in the population
        initial_value = network.populations[pop]['initial_value']
        for num in range(num_neurons_in_pop):  # for each neuron, copy the parameters over
            c_m[index] = network.populations[pop]['type'].params['membrane_capacitance']
            g_m[index] = network.populations[pop]['type'].params['membrane_conductance']
            u_rest[index] = network.populations[pop]['type'].params['resting_potential']

            i_b = i_b.to_dense()
            i_b[0, index] = network.populations[pop]['type'].params['bias']
            i_b = i_b.to_sparse()

            if hasattr(initial_value, '__iter__'):
                u_last[index] = initial_value[num]
            elif initial_value is None:
                u_last[index] = u_rest[index]
            else:
                u_last[index] = initial_value

            if spiking:
                if isinstance(network.populations[pop]['type'],
                              SpikingNeuron):  # if the neuron is spiking, copy more
                    theta_0[index] = network.populations[pop]['type'].params['threshold_initial_value']

                    m = m.to_dense()
                    m[0, index] = network.populations[pop]['type'].params[
                        'threshold_proportionality_constant']
                    m = m.to_sparse()

                    tau_theta[index] = network.populations[pop]['type'].params['threshold_time_constant']
                else:  # otherwise, set to the special values for NonSpiking
                    theta_0[index] = torch.finfo(theta_0[index].dtype).max

                    m = m.to_dense()
                    m[0, index] = 0
                    m = m.to_sparse()

                    tau_theta[index] = 1
            if gated:
                if isinstance(network.populations[pop]['type'], NonSpikingNeuronWithGatedChannels):
                    # Channel params
                    g_ion = g_ion.to_dense()
                    g_ion[:, index] = network.populations[pop]['type'].params['Gion']
                    g_ion = g_ion.to_sparse()

                    e_ion = e_ion.to_dense()
                    e_ion[:, index] = network.populations[pop]['type'].params['Eion']
                    e_ion = e_ion.to_sparse()

                    # A gate params
                    pow_a = pow_a.to_dense()
                    pow_a[:, index] = network.populations[pop]['type'].params['paramsA']['pow']
                    pow_a = pow_a.to_sparse()

                    slope_a = slope_a.to_dense()
                    slope_a[:, index] = network.populations[pop]['type'].params['paramsA']['slope']
                    slope_a = slope_a.to_sparse()

                    k_a[:, index] = network.populations[pop]['type'].params['paramsA']['k']

                    e_a = e_a.to_dense()
                    e_a[:, index] = network.populations[pop]['type'].params['paramsA']['reversal']
                    e_a = e_a.to_sparse()

                    # B gate params
                    pow_b = pow_b.to_dense()
                    pow_b[:, index] = network.populations[pop]['type'].params['paramsB']['pow']
                    pow_b = pow_b.to_sparse()

                    slope_b = slope_b.to_dense()
                    slope_b[:, index] = network.populations[pop]['type'].params['paramsB']['slope']
                    slope_b = slope_b.to_sparse()

                    k_b[:, index] = network.populations[pop]['type'].params['paramsB']['k']

                    e_b = e_b.to_dense()
                    e_b[:, index] = network.populations[pop]['type'].params['paramsB']['reversal']
                    e_b = e_b.to_sparse()

                    tau_max_b[:, index] = network.populations[pop]['type'].params['paramsB']['TauMax']

                    # C gate params
                    pow_c = pow_c.to_dense()
                    pow_c[:, index] = network.populations[pop]['type'].params['paramsC']['pow']
                    pow_c = pow_c.to_sparse()

                    slope_c = slope_c.to_dense()
                    slope_c[:, index] = network.populations[pop]['type'].params['paramsC']['slope']
                    slope_c = slope_c.to_sparse()

                    k_c[:, index] = network.populations[pop]['type'].params['paramsC']['k']

                    e_c = e_c.to_dense()
                    e_c[:, index] = network.populations[pop]['type'].params['paramsC']['reversal']
                    e_c = e_c.to_sparse()

                    tau_max_c[:, index] = network.populations[pop]['type'].params['paramsC']['TauMax']

                    b_gate_last = b_gate_last.to_dense()
                    b_gate_last[:, index] = 1 / (1 + k_b[:, index] * np.exp(
                        slope_b.to_dense()[:, index] * (u_last[index] - e_b.to_dense()[:, index])))
                    b_gate_last = b_gate_last.to_sparse()

                    c_gate_last = c_gate_last.to_dense()
                    c_gate_last[:, index] = 1 / (1 + k_c[:, index] * np.exp(
                        slope_c.to_dense()[:, index] * (u_last[index] - e_c.to_dense()[:, index])))
                    c_gate_last = c_gate_last.to_sparse()

            index += 1
    u = u_last.clone()
    if spiking:
        theta = theta_0.clone()
        theta_last = theta_0.clone()
    if gated:
        b_gate = torch.clone(b_gate_last)
        b_gate_0 = torch.clone(b_gate_last)
        c_gate = torch.clone(c_gate_last)
        c_gate_0 = torch.clone(c_gate_last)

    """
    --------------------------------------------------------------------------------------------------------------------
    Set Inputs
    --------------------------------------------------------------------------------------------------------------------
    """
    if debug:
        print('--------------')
        print('Setting Inputs')
        print('--------------')
    input_connectivity = torch.sparse_coo_tensor(size=(num_neurons, network.get_num_inputs_actual()),
                                                      device=device)  # initialize connectivity matrix
    index = 0
    for inp in range(network.get_num_inputs()):  # iterate over the connections in the network
        size = network.inputs[inp]['size']
        dest_pop = network.inputs[inp]['destination']  # get the destination

        input_connectivity = input_connectivity.to_dense()
        if size == 1:
            for dest in pops_and_nrns[dest_pop]:
                input_connectivity[dest][inp] = 1.0  # set the weight in the correct source and destination
            index += 1
        else:
            for dest in pops_and_nrns[dest_pop]:
                input_connectivity[dest][index] = 1.0
                index += 1
        input_connectivity = input_connectivity.to_sparse()

    """
    --------------------------------------------------------------------------------------------------------------------
    Set Connections
    --------------------------------------------------------------------------------------------------------------------
    """
    if debug:
        print('-------------------')
        print('Setting connections')
        print('-------------------')
    for syn in range(len(network.connections)):
        source_pop = network.connections[syn]['source']
        dest_pop = network.connections[syn]['destination']
        g_max_val = network.connections[syn]['params']['max_conductance']
        if network.connections[syn]['params']['electrical'] is False:  # chemical connection
            del_e_val = network.connections[syn]['params']['reversal_potential']

        if network.connections[syn]['params']['matrix']:  # pattern and matrix connections
            pop_size_source = len(pops_and_nrns[source_pop])
            pop_size_dest = len(pops_and_nrns[dest_pop])
            source_index = pops_and_nrns[source_pop][0]
            dest_index = pops_and_nrns[dest_pop][0]

            if network.connections[syn]['params']['spiking']:
                tau_s = network.connections[syn]['params']['synapticTimeConstant']

                g_max_spike = g_max_spike.to_dense()
                g_max_spike[dest_index:dest_index + pop_size_dest,
                source_index:source_index + pop_size_source] = torch.from_numpy(g_max_val)
                g_max_spike = g_max_spike.to_sparse()

                del_e = del_e.to_dense()
                del_e[dest_index:dest_index + pop_size_dest, source_index:source_index + pop_size_source] = torch.from_numpy(
                    del_e_val)
                del_e = del_e.to_sparse()

                tau_syn[dest_index:dest_index + pop_size_dest, source_index:source_index + pop_size_source] = torch.from_numpy(
                    tau_s)

                if delay:
                    delay_val = network.connections[syn]['params']['synapticTransmissionDelay']
                    spike_delays = spike_delays.to_dense()
                    spike_delays[dest_index:dest_index + pop_size_dest,
                    source_index:source_index + pop_size_source] = torch.from_numpy(delay_val)
                    spike_delays = spike_delays.to_sparse()

                    for source in pops_and_nrns[source_pop]:
                        for dest in pops_and_nrns[dest_pop]:
                            buffer_nrns.append(source)
                            buffer_steps.append(delay_val)
                            spike_rows.append(dest)
                            spike_cols.append(source)
            else:
                e_lo_val = network.connections[syn]['params']['e_lo']
                e_hi_val = network.connections[syn]['params']['e_hi']

                g_max_non = g_max_non.to_dense()
                g_max_non[dest_index:dest_index + pop_size_dest,
                source_index:source_index + pop_size_source] = torch.from_numpy(g_max_val)
                g_max_non = g_max_non.to_sparse()

                del_e = del_e.to_dense()
                del_e[dest_index:dest_index + pop_size_dest, source_index:source_index + pop_size_source] = torch.from_numpy(
                    del_e_val)
                del_e = del_e.to_sparse()

                e_lo = e_lo.to_dense()
                e_lo[dest_index:dest_index + pop_size_dest, source_index:source_index + pop_size_source] = torch.from_numpy(
                    e_lo_val)
                e_lo = e_lo.to_sparse()

                e_hi[dest_index:dest_index + pop_size_dest, source_index:source_index + pop_size_source] = torch.from_numpy(
                    e_hi_val)
        elif network.connections[syn]['params']['electrical']:  # electrical connection
            for source in pops_and_nrns[source_pop]:
                for dest in pops_and_nrns[dest_pop]:
                    if network.connections[syn]['params']['rectified']:  # rectified
                        g_rectified = g_rectified.to_dense()
                        g_rectified[dest][source] = g_max_val / len(pops_and_nrns[source_pop])
                        g_rectified = g_rectified.to_sparse()
                    else:
                        g_electrical = g_electrical.to_dense()
                        g_electrical[dest][source] = g_max_val / len(pops_and_nrns[source_pop])
                        g_electrical[source][dest] = g_max_val / len(pops_and_nrns[source_pop])
                        g_electrical = g_electrical.to_sparse()
        else:  # chemical connections
            if network.connections[syn]['params']['spiking']:  # spiking chemical synapse
                tau_s = network.connections[syn]['params']['synapticTimeConstant']
                if delay:
                    delay_val = network.connections[syn]['params']['synapticTransmissionDelay']
                for source in pops_and_nrns[source_pop]:
                    for dest in pops_and_nrns[dest_pop]:
                        g_max_spike = g_max_spike.to_dense()
                        g_max_spike[dest][source] = g_max_val / len(pops_and_nrns[source_pop])
                        g_max_spike = g_max_spike.to_sparse()

                        del_e = del_e.to_dense()
                        del_e[dest][source] = del_e_val
                        del_e = del_e.to_sparse()

                        tau_syn[dest][source] = tau_s

                        if delay:
                            spike_delays = spike_delays.to_dense()
                            spike_delays[dest][source] = delay_val
                            spike_delays = spike_delays.to_sparse()

                            buffer_nrns.append(source)
                            buffer_steps.append(delay_val)
                            spike_rows.append(dest)
                            spike_cols.append(source)
            else:  # non-spiking chemical synapse
                e_lo_val = network.connections[syn]['params']['e_lo']
                e_hi_val = network.connections[syn]['params']['e_hi']
                for source in pops_and_nrns[source_pop]:
                    for dest in pops_and_nrns[dest_pop]:
                        g_max_non = g_max_non.to_dense()
                        g_max_non[dest][source] = g_max_val / len(pops_and_nrns[source_pop])
                        g_max_non = g_max_non.to_sparse()

                        del_e = del_e.to_dense()
                        del_e[dest][source] = del_e_val
                        del_e = del_e.to_sparse()

                        e_lo = e_lo.to_dense()
                        e_lo[dest][source] = e_lo_val
                        e_lo = e_lo.to_sparse()

                        e_hi[dest][source] = e_hi_val

    """
    --------------------------------------------------------------------------------------------------------------------
    Calculate Time Factors
    --------------------------------------------------------------------------------------------------------------------
    """
    if debug:
        print('------------------------')
        print('Calculating Time Factors')
        print('------------------------')
    time_factor_membrane = dt / (c_m / g_m)
    if spiking:
        time_factor_threshold = dt / tau_theta
        time_factor_synapse = dt / tau_syn

    """
    --------------------------------------------------------------------------------------------------------------------
    Initialize Propagation Delay
    --------------------------------------------------------------------------------------------------------------------
    """
    if delay:
        if debug:
            print('------------------------------')
            print('Initializing Propagation Delay')
            print('------------------------------')
        spike_delays = spike_delays.to_dense()
        buffer_length = int(torch.max(spike_delays) + 1)
        spike_delays = spike_delays.to_sparse()

        spike_buffer = torch.sparse_coo_tensor(size=(buffer_length, num_neurons), device=device)

    """
    --------------------------------------------------------------------------------------------------------------------
    Set Outputs
    --------------------------------------------------------------------------------------------------------------------
    """
    if debug:
        print('---------------')
        print('Setting Outputs')
        print('---------------')
    output_nodes = []
    index = 0
    for out in range(len(network.outputs)):
        source_pop = network.outputs[out]['source']
        num_source_neurons = network.populations[source_pop]['number']
        output_nodes.append([])
        for num in range(num_source_neurons):
            output_nodes[out].append(index)
            index += 1
    num_outputs = index

    output_voltage_connectivity = torch.sparse_coo_tensor(size=(num_outputs, num_neurons),
                                                               device=device)  # initialize connectivity matrix
    if spiking:
        output_spike_connectivity = torch.clone(output_voltage_connectivity)
    outputs = torch.sparse_coo_tensor(size=(1, num_outputs), device=device)

    for out in range(len(network.outputs)):  # iterate over the connections in the network
        source_pop = network.outputs[out]['source']  # get the source
        for i in range(len(pops_and_nrns[source_pop])):
            if network.outputs[out]['spiking']:
                output_spike_connectivity = output_spike_connectivity.to_dense()
                output_spike_connectivity[output_nodes[out][i]][
                    pops_and_nrns[source_pop][i]] = 1.0  # set the weight in the correct source and destination
                output_spike_connectivity = output_spike_connectivity.to_sparse()
            else:
                output_voltage_connectivity = output_voltage_connectivity.to_dense()
                output_voltage_connectivity[output_nodes[out][i]][
                    pops_and_nrns[source_pop][i]] = 1.0  # set the weight in the correct source and destination
                output_voltage_connectivity = output_voltage_connectivity.to_sparse()

    """
    --------------------------------------------------------------------------------------------------------------------
    Arrange states and parameters into dictionary
    --------------------------------------------------------------------------------------------------------------------
    """
    if debug:
        print('---------------------')
        print('Writing to Dictionary')
        print('---------------------')
    params = {'dt': dt,
              'name': name,
              'spiking': spiking,
              'delay': delay,
              'elec': electrical,
              'rect': electrical_rectified,
              'gated': gated,
              'numChannels': num_channels,
              'v': u,
              'vLast': u_last,
              'vRest': u_rest,
              'v0': u_0,
              'cM': c_m,
              'gM': g_m,
              'iB': i_b,
              'gMaxNon': g_max_non,
              'delE': del_e,
              'eLo': e_lo,
              'eHi': e_hi,
              'timeFactorMembrane': time_factor_membrane,
              'inputConn': input_connectivity,
              'numPop': num_populations,
              'numNeurons': num_neurons,
              'numConn': num_connections,
              'numInputs': num_inputs,
              'numOutputs': num_outputs,
              # 'r': R,
              'outConnVolt': output_voltage_connectivity}
    if spiking:
        params['spikes'] = spikes
        params['theta0'] = theta_0
        params['theta'] = theta
        params['thetaLast'] = theta_last
        params['m'] = m
        params['tauTheta'] = tau_theta
        params['gMaxSpike'] = g_max_spike
        params['gSpike'] = g_spike
        params['tauSyn'] = tau_syn
        params['timeFactorThreshold'] = time_factor_threshold
        params['timeFactorSynapse'] = time_factor_synapse
        params['outConnSpike'] = output_spike_connectivity
    if delay:
        params['spikeDelays'] = spike_delays
        params['spikeRows'] = spike_rows
        params['spikeCols'] = spike_cols
        params['bufferSteps'] = buffer_steps
        params['bufferNrns'] = buffer_nrns
        params['delayedSpikes'] = delayed_spikes
        params['spikeBuffer'] = spike_buffer
    if electrical:
        params['gElectrical'] = g_electrical
    if electrical_rectified:
        params['gRectified'] = g_rectified
    if gated:
        params['gIon'] = g_ion
        params['eIon'] = e_ion
        params['powA'] = pow_a
        params['slopeA'] = slope_a
        params['kA'] = k_a
        params['eA'] = e_a
        params['powB'] = pow_b
        params['slopeB'] = slope_b
        params['kB'] = k_b
        params['eB'] = e_b
        params['tauMaxB'] = tau_max_b
        params['powC'] = pow_c
        params['slopeC'] = slope_c
        params['kC'] = k_c
        params['eC'] = e_c
        params['tauMaxC'] = tau_max_c
        params['bGate'] = b_gate
        params['bGateLast'] = b_gate_last
        params['bGate0'] = b_gate_0
        params['cGate'] = c_gate
        params['cGateLast'] = c_gate_last
        params['cGate0'] = c_gate_0

    """
    --------------------------------------------------------------------------------------------------------------------
    Passing params to backend object
    --------------------------------------------------------------------------------------------------------------------
    """
    if debug:
        print('-------------------------------------------------')
        print('Passing states and parameters to SNS_Numpy object')
        print('-------------------------------------------------')
    model = SNS_Sparse(params)

    """
    --------------------------------------------------------------------------------------------------------------------
    Final print
    --------------------------------------------------------------------------------------------------------------------
    """
    if debug:
        print('----------------------------')
        print('Final states and parameters:')
        print('----------------------------')
        print('Input Connectivity:')
        print(input_connectivity)
        print('g_max_non:')
        print(g_max_non)
        if spiking:
            print('GmaxSpike:')
            print(g_max_spike)
        print('del_e:')
        print(del_e)
        print('e_lo:')
        print(e_lo)
        print('e_hi:')
        print(e_hi)
        if electrical:
            print('Gelectrical:')
            print(g_electrical)
        if electrical_rectified:
            print('GelectricalRectified:')
            print(g_rectified)
        print('Output Voltage Connectivity')
        print(output_voltage_connectivity)
        if spiking:
            print('Output Spike Connectivity:')
            print(output_spike_connectivity)
        print('v:')
        print(u)
        print('v_last:')
        print(u_last)
        print('v_rest:')
        print(u_rest)
        if spiking:
            print('theta_0:')
            print(theta_0)
            print('ThetaLast:')
            print(theta_last)
            print('Theta')
            print(theta)
        if gated:
            print('Number of Channels:')
            print(num_channels)
            print('Ionic Conductance:')
            print(g_ion)
            print('Ionic Reversal Potentials:')
            print(e_ion)
            print('A Gate Parameters:')
            print('Power:')
            print(pow_a)
            print('Slope:')
            print(slope_a)
            print('K:')
            print(k_a)
            print('Reversal Potential:')
            print(e_a)
            print('B Gate Parameters:')
            print('Power:')
            print(pow_b)
            print('Slope:')
            print(slope_b)
            print('K:')
            print(k_b)
            print('Reversal Potential:')
            print(e_b)
            print('Tau Max:')
            print(tau_max_b)
            print('B:')
            print(b_gate)
            print('B_last:')
            print(b_gate_last)
            print('C Gate Parameters:')
            print('Power:')
            print(pow_c)
            print('Slope:')
            print(slope_c)
            print('K:')
            print(k_c)
            print('Reversal Potential:')
            print(e_c)
            print('Tau Max:')
            print(tau_max_c)
            print('B:')
            print(c_gate)
            print('B_last:')
            print(c_gate_last)

    return model

def __compile_manual__(network, dt=0.01, debug=False) -> SNS_Iterative:
    if debug:
        print('-------------------------------------------------------------------------------------------------------')
        print('COMPILING NETWORK USING NUMPY (ITERATIVE):')
        print('-------------------------------------------------------------------------------------------------------')

    """
    --------------------------------------------------------------------------------------------------------------------
    Get net parameters
    --------------------------------------------------------------------------------------------------------------------
    """
    if debug:
        print('-----------------------------')
        print('Getting network parameters...')
        print('-----------------------------')
    spiking = network.params['spiking']
    delay = network.params['delay']
    electrical = network.params['electrical']
    electrical_rectified = network.params['electricalRectified']
    gated = network.params['gated']
    num_channels = network.params['numChannels']
    name = network.params['name']
    num_populations = network.get_num_populations()
    num_neurons = network.get_num_neurons()
    num_connections = network.get_num_connections()
    num_inputs = network.get_num_inputs()
    num_outputs = network.get_num_outputs()
    # R = network.params['R']
    if debug:
        print('Spiking:')
        print(spiking)
        print('Spiking Propagation Delay:')
        print(delay)
        print('Electrical Synapses:')
        print(electrical)
        print('Rectified Electrical Synapses:')
        print(electrical_rectified)
        print('Number of Populations:')
        print(num_populations)
        print('Number of Neurons:')
        print(num_neurons)
        print('Number of Connections')
        print(num_connections)
        print('Number of Inputs:')
        print(num_inputs)
        print('Number of Outputs:')
        print(num_outputs)
        # print('Network Voltage Range (mV):')
        # print(R)

    """
    --------------------------------------------------------------------------------------------------------------------
    Initialize vectors and matrices
    --------------------------------------------------------------------------------------------------------------------
    """
    if debug:
        print('---------------------------------')
        print('Initializing vectors and matrices')
        print('---------------------------------')
    V = np.zeros(num_neurons)
    V_0 = np.zeros(num_neurons)
    V_last = np.zeros(num_neurons)
    V_rest = np.zeros(num_neurons)
    c_m = np.zeros(num_neurons)
    g_m = np.zeros(num_neurons)
    i_b = np.zeros(num_neurons)
    if spiking:
        spikes = np.zeros(num_neurons)
        theta_0 = np.zeros(num_neurons)
        theta = np.zeros(num_neurons)
        theta_last = np.zeros(num_neurons)
        m = np.zeros(num_neurons)
        tau_theta = np.zeros(num_neurons)
    if gated:
        # Channel params
        g_ion = np.zeros([num_channels, num_neurons])
        e_ion = np.zeros([num_channels, num_neurons])
        # A gate params
        pow_a = np.zeros([num_channels, num_neurons])
        slope_a = np.zeros([num_channels, num_neurons])
        k_a = np.zeros([num_channels, num_neurons]) + 1
        e_a = np.zeros([num_channels, num_neurons])
        # B gate params
        pow_b = np.zeros([num_channels, num_neurons])
        slope_b = np.zeros([num_channels, num_neurons])
        k_b = np.zeros([num_channels, num_neurons]) + 1
        e_b = np.zeros([num_channels, num_neurons])
        tau_max_b = np.zeros([num_channels, num_neurons]) + 1
        # C gate params
        pow_c = np.zeros([num_channels, num_neurons])
        slope_c = np.zeros([num_channels, num_neurons])
        k_c = np.zeros([num_channels, num_neurons]) + 1
        e_c = np.zeros([num_channels, num_neurons])
        tau_max_c = np.zeros([num_channels, num_neurons]) + 1

        b_gate = np.zeros([num_channels, num_neurons])
        b_gate_0 = np.zeros([num_channels, num_neurons])
        b_gate_last = np.zeros([num_channels, num_neurons])
        c_gate = np.zeros([num_channels, num_neurons])
        c_gate_0 = np.zeros([num_channels, num_neurons])
        c_gate_last = np.zeros([num_channels, num_neurons])

    incoming_synapses = []
    for i in range(num_neurons):
        incoming_synapses.append([])

    pops_and_nrns = []
    index = 0
    for pop in range(len(network.populations)):
        num_neurons_in_pop = network.populations[pop]['number']  # find the number of neurons in the population
        pops_and_nrns.append([])
        for num in range(num_neurons_in_pop):
            pops_and_nrns[pop].append(index)
            index += 1

    """
    --------------------------------------------------------------------------------------------------------------------
    Set Neurons
    --------------------------------------------------------------------------------------------------------------------
    """
    if debug:
        print('---------------')
        print('Setting neurons')
        print('---------------')
    index = 0
    for pop in range(len(network.populations)):
        num_neurons_in_pop = network.populations[pop]['number']  # find the number of neurons in the population
        initial_value = network.populations[pop]['initial_value']
        for num in range(num_neurons_in_pop):  # for each neuron, copy the parameters over
            c_m[index] = network.populations[pop]['type'].params['membrane_capacitance']
            g_m[index] = network.populations[pop]['type'].params['membrane_conductance']
            i_b[index] = network.populations[pop]['type'].params['bias']
            V_rest[index] = network.populations[pop]['type'].params['resting_potential']
            if hasattr(initial_value, '__iter__'):
                V_last[index] = initial_value[num]
            elif initial_value is None:
                V_last[index] = 0.0
            else:
                V_last[index] = initial_value
            if spiking:
                if isinstance(network.populations[pop]['type'],
                              SpikingNeuron):  # if the neuron is spiking, copy more
                    theta_0[index] = network.populations[pop]['type'].params['threshold_initial_value']
                    m[index] = network.populations[pop]['type'].params['threshold_proportionality_constant']
                    tau_theta[index] = network.populations[pop]['type'].params['threshold_time_constant']
                else:  # otherwise, set to the special values for NonSpiking
                    theta_0[index] = sys.float_info.max
                    m[index] = 0
                    tau_theta[index] = 1
            if gated:
                if isinstance(network.populations[pop]['type'], NonSpikingNeuronWithGatedChannels):
                    # Channel params
                    g_ion[:, index] = network.populations[pop]['type'].params['Gion']
                    e_ion[:, index] = network.populations[pop]['type'].params['Eion']
                    # A gate params
                    pow_a[:, index] = network.populations[pop]['type'].params['paramsA']['pow']
                    slope_a[:, index] = network.populations[pop]['type'].params['paramsA']['slope']
                    k_a[:, index] = network.populations[pop]['type'].params['paramsA']['k']
                    e_a[:, index] = network.populations[pop]['type'].params['paramsA']['reversal']
                    # B gate params
                    pow_b[:, index] = network.populations[pop]['type'].params['paramsB']['pow']
                    slope_b[:, index] = network.populations[pop]['type'].params['paramsB']['slope']
                    k_b[:, index] = network.populations[pop]['type'].params['paramsB']['k']
                    e_b[:, index] = network.populations[pop]['type'].params['paramsB']['reversal']
                    tau_max_b[:, index] = network.populations[pop]['type'].params['paramsB']['TauMax']
                    # C gate params
                    pow_c[:, index] = network.populations[pop]['type'].params['paramsC']['pow']
                    slope_c[:, index] = network.populations[pop]['type'].params['paramsC']['slope']
                    k_c[:, index] = network.populations[pop]['type'].params['paramsC']['k']
                    e_c[:, index] = network.populations[pop]['type'].params['paramsC']['reversal']
                    tau_max_c[:, index] = network.populations[pop]['type'].params['paramsC']['TauMax']

                    b_gate_last[:, index] = 1 / (1 + k_b[:, index] * np.exp(
                        slope_b[:, index] * (V_last[index] - e_b[:, index])))
                    c_gate_last[:, index] = 1 / (1 + k_c[:, index] * np.exp(
                        slope_c[:, index] * (V_last[index] - e_c[:, index])))
            index += 1
    V = np.copy(V_last)
    if spiking:
        theta = np.copy(theta_0)
        theta_last = np.copy(theta_0)
    if gated:
        b_gate = np.copy(b_gate_last)
        b_gate_0 = np.copy(b_gate_last)
        c_gate = np.copy(c_gate_last)
        c_gate_0 = np.copy(c_gate_last)

    """
    --------------------------------------------------------------------------------------------------------------------
    Set Inputs
    --------------------------------------------------------------------------------------------------------------------
    """
    if debug:
        print('--------------')
        print('Setting Inputs')
        print('--------------')
    input_connectivity = np.zeros(
        [num_neurons, network.get_num_inputs_actual()])  # initialize connectivity matrix
    index = 0
    for inp in range(network.get_num_inputs()):  # iterate over the connections in the network
        size = network.inputs[inp]['size']
        dest_pop = network.inputs[inp]['destination']  # get the destination
        if size == 1:
            for dest in pops_and_nrns[dest_pop]:
                input_connectivity[dest][inp] = 1.0  # set the weight in the correct source and destination
            index += 1
        else:
            for dest in pops_and_nrns[dest_pop]:
                input_connectivity[dest][index] = 1.0
                index += 1

    """
    --------------------------------------------------------------------------------------------------------------------
    Set Connections
    --------------------------------------------------------------------------------------------------------------------
    """
    if debug:
        print('-------------------')
        print('Setting connections')
        print('-------------------')
    for syn in range(len(network.connections)):
        source_pop = network.connections[syn]['source']
        dest_pop = network.connections[syn]['destination']
        g_max_val = network.connections[syn]['params']['max_conductance']
        if network.connections[syn]['params']['electrical'] is False:  # electrical connection
            del_e_val = network.connections[syn]['params']['reversal_potential']

        if network.connections[syn]['params']['matrix']:  # pattern and matrix connections
            pop_size_source = len(pops_and_nrns[source_pop])
            pop_size_dest = len(pops_and_nrns[dest_pop])
            source_index = pops_and_nrns[source_pop][0]
            dest_index = pops_and_nrns[dest_pop][0]
            if network.connections[syn]['params']['spiking']:
                tau_s = network.connections[syn]['params']['synapticTimeConstant']
                if delay:
                    delay_val = network.connections[syn]['params']['synapticTransmissionDelay']

                for dest in range(pop_size_dest):
                    for source in range(pop_size_source):
                        g_syn = g_max_val[dest, source]
                        rev = del_e_val[dest, source]
                        time_factor_syn = dt / tau_s[dest, source]
                        if delay:
                            buffer = np.zeros(delay_val[dest, source])
                            incoming_synapses[dest + dest_index].append(
                                [source + source_index, True, False, g_syn, rev, 0, time_factor_syn, buffer])
                        else:
                            incoming_synapses[dest + dest_index].append(
                                [source + source_index, True, False, g_syn, rev, 0, time_factor_syn])
            else:
                e_lo_val = network.connections[syn]['params']['e_lo']
                e_hi_val = network.connections[syn]['params']['e_hi']
                for dest in range(pop_size_dest):
                    for source in range(pop_size_source):
                        g_syn = g_max_val[dest, source]
                        rev = del_e_val[dest, source]

                        e_hi = e_hi_val[dest, source]
                        e_lo = e_lo_val[dest, source]

                        incoming_synapses[dest + dest_index].append(
                            [source + source_index, False, False, g_syn, rev, 0, e_lo, e_hi])
        elif network.connections[syn]['params']['electrical']:  # electrical connection
            for dest in pops_and_nrns[dest_pop]:
                for source in pops_and_nrns[source_pop]:
                    g_syn = g_max_val / len(pops_and_nrns[source_pop])
                    if network.connections[syn]['params']['rectified']:  # rectified
                        incoming_synapses[dest].append([source, False, True, g_syn, True, source, dest])
                        incoming_synapses[source].append([dest, False, True, g_syn, True, source, dest])
                    else:
                        incoming_synapses[dest].append([source, False, True, g_syn, False])
                        incoming_synapses[source].append([dest, False, True, g_syn, False])
        else:  # chemical connection
            if network.connections[syn]['params']['spiking']:
                tau_s = network.connections[syn]['params']['synapticTimeConstant']
                if delay:
                    delay_val = network.connections[syn]['params']['synapticTransmissionDelay']
                for dest in pops_and_nrns[dest_pop]:
                    for source in pops_and_nrns[source_pop]:
                        g_syn = g_max_val / len(pops_and_nrns[source_pop])
                        if delay:
                            buffer = np.zeros(delay_val + 1)
                            incoming_synapses[dest].append(
                                [source, True, False, g_syn, del_e_val, 0, dt / tau_s, buffer])
                        else:
                            incoming_synapses[dest].append([source, True, False, g_syn, del_e_val, 0, dt / tau_s])
            else:
                e_lo_val = network.connections[syn]['params']['e_lo']
                e_hi_val = network.connections[syn]['params']['e_hi']
                for dest in pops_and_nrns[dest_pop]:
                    for source in pops_and_nrns[source_pop]:
                        g_syn = g_max_val / len(pops_and_nrns[source_pop])

                        incoming_synapses[dest].append([source, False, False, g_syn, del_e_val, 0, e_lo_val, e_hi_val])

    """
    --------------------------------------------------------------------------------------------------------------------
    Calculate Time Factors
    --------------------------------------------------------------------------------------------------------------------
    """
    if debug:
        print('------------------------')
        print('Calculating Time Factors')
        print('------------------------')
    time_factor_membrane = dt / (c_m / g_m)
    if spiking:
        time_factor_threshold = dt / tau_theta

    
    """
    --------------------------------------------------------------------------------------------------------------------
    Set Outputs
    --------------------------------------------------------------------------------------------------------------------
    """
    if debug:
        print('---------------')
        print('Setting Outputs')
        print('---------------')
    output_nodes = []
    index = 0
    for out in range(len(network.outputs)):
        source_pop = network.outputs[out]['source']
        num_source_neurons = network.populations[source_pop]['number']
        output_nodes.append([])
        for num in range(num_source_neurons):
            output_nodes[out].append(index)
            index += 1
    num_outputs = index

    output_voltage_connectivity = np.zeros([num_outputs, num_neurons])  # initialize connectivity matrix
    if spiking:
        output_spike_connectivity = np.copy(output_voltage_connectivity)
    outputs = np.zeros(num_outputs)
    for out in range(len(network.outputs)):  # iterate over the connections in the network
        source_pop = network.outputs[out]['source']  # get the source
        for i in range(len(pops_and_nrns[source_pop])):
            if network.outputs[out]['spiking']:
                output_spike_connectivity[output_nodes[out][i]][
                    pops_and_nrns[source_pop][i]] = 1.0  # set the weight in the correct source and destination
                # out_linear[outputs[out][i]] = 1.0
            else:
                output_voltage_connectivity[output_nodes[out][i]][
                    pops_and_nrns[source_pop][i]] = 1.0  # set the weight in the correct source and destination

    """
    --------------------------------------------------------------------------------------------------------------------
    Arrange states and parameters into dictionary
    --------------------------------------------------------------------------------------------------------------------
    """
    if debug:
        print('---------------------')
        print('Writing to Dictionary')
        print('---------------------')
    params = {'dt': dt,
            'name': name,
            'spiking': spiking,
            'delay': delay,
            'elec': electrical,
            'rect': electrical_rectified,
            'gated': gated,
            'numChannels': num_channels,
            'v': V,
            'vLast': V_last,
            'vRest': V_rest,
            'v0': V_0,
            'cM': c_m,
            'gM': g_m,
            'iB': i_b,
            'timeFactorMembrane': time_factor_membrane,
            'inputConn': input_connectivity,
            'numPop': num_populations,
            'numNeurons': num_neurons,
            'numConn': num_connections,
            'numInputs': num_inputs,
            'numOutputs': num_outputs,
            # 'r': R,
            'outConnVolt': output_voltage_connectivity,
            'incomingSynapses': incoming_synapses}
    if spiking:
        params['spikes'] = spikes
        params['theta0'] = theta_0
        params['theta'] = theta
        params['thetaLast'] = theta_last
        params['m'] = m
        params['tauTheta'] = tau_theta
        params['timeFactorThreshold'] = time_factor_threshold
        params['outConnSpike'] = output_spike_connectivity
    if gated:
        params['gIon'] = g_ion
        params['eIon'] = e_ion
        params['powA'] = pow_a
        params['slopeA'] = slope_a
        params['kA'] = k_a
        params['eA'] = e_a
        params['powB'] = pow_b
        params['slopeB'] = slope_b
        params['kB'] = k_b
        params['eB'] = e_b
        params['tauMaxB'] = tau_max_b
        params['powC'] = pow_c
        params['slopeC'] = slope_c
        params['kC'] = k_c
        params['eC'] = e_c
        params['tauMaxC'] = tau_max_c
        params['bGate'] = b_gate
        params['bGateLast'] = b_gate_last
        params['bGate0'] = b_gate_0
        params['cGate'] = c_gate
        params['cGateLast'] = c_gate_last
        params['cGate0'] = c_gate_0

    """
    --------------------------------------------------------------------------------------------------------------------
    Passing params to backend object
    --------------------------------------------------------------------------------------------------------------------
    """
    if debug:
        print('-------------------------------------------------')
        print('Passing states and parameters to SNS_Numpy object')
        print('-------------------------------------------------')
    model = SNS_Iterative(params)

    """
    --------------------------------------------------------------------------------------------------------------------
    Final print
    --------------------------------------------------------------------------------------------------------------------
    """
    if debug:
        print('----------------------------')
        print('Final states and parameters:')
        print('----------------------------')
    print('Input Connectivity:')
    print(input_connectivity)
    print('Output Voltage Connectivity')
    print(output_voltage_connectivity)
    if spiking:
        print('Output Spike Connectivity:')
        print(output_spike_connectivity)
    print('v:')
    print(V)
    print('v_last:')
    print(V_last)
    print('v_rest:')
    print(V_rest)
    if spiking:
        print('theta_0:')
        print(theta_0)
        print('ThetaLast:')
        print(theta_last)
        print('Theta')
        print(theta)
    if gated:
        print('Number of Channels:')
        print(num_channels)
        print('Ionic Conductance:')
        print(g_ion)
        print('Ionic Reversal Potentials:')
        print(e_ion)
        print('A Gate Parameters:')
        print('Power:')
        print(pow_a)
        print('Slope:')
        print(slope_a)
        print('K:')
        print(k_a)
        print('Reversal Potential:')
        print(e_a)
        print('B Gate Parameters:')
        print('Power:')
        print(pow_b)
        print('Slope:')
        print(slope_b)
        print('K:')
        print(k_b)
        print('Reversal Potential:')
        print(e_b)
        print('Tau Max:')
        print(tau_max_b)
        print('B:')
        print(b_gate)
        print('B_last:')
        print(b_gate_last)
        print('C Gate Parameters:')
        print('Power:')
        print(pow_c)
        print('Slope:')
        print(slope_c)
        print('K:')
        print(k_c)
        print('Reversal Potential:')
        print(e_c)
        print('Tau Max:')
        print(tau_max_c)
        print('B:')
        print(c_gate)
        print('B_last:')
        print(c_gate_last)

    return model
