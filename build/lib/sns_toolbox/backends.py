"""
Simulation backends for synthetic nervous system networks. Each of these are python-based, and are constructed using a
Network. They can then be run for a step, with the inputs being a vector of neural states and applied currents and the
output being the next step of neural states.
"""

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
IMPORTS
"""

from typing import Dict
import numpy as np
import torch

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
BACKENDS
"""

class Backend:

    def __init__(self, params: Dict) -> None:
        self.set_params(params)

    def forward(self, x=None):
        raise NotImplementedError

    def set_params(self, params: Dict) -> None:
        self.dt = params['dt']
        self.name = params['name']
        self.spiking = params['spiking']
        self.delay = params['delay']
        self.electrical = params['elec']
        self.electrical_rectified = params['rect']
        self.gated = params['gated']
        self.num_channels = params['numChannels']
        self.V = params['v']
        self.V_last = params['vLast']
        self.V_0 = params['v0']
        self.V_rest = params['vRest']
        self.c_m = params['cM']
        self.g_m = params['gM']
        self.i_b = params['iB']
        self.g_max_non = params['gMaxNon']
        self.del_e = params['delE']
        self.e_lo = params['eLo']
        self.e_hi = params['eHi']
        self.time_factor_membrane = params['timeFactorMembrane']
        self.input_connectivity = params['inputConn']
        self.output_voltage_connectivity = params['outConnVolt']
        self.num_populations = params['numPop']
        self.num_neurons = params['numNeurons']
        self.num_connections = params['numConn']
        self.num_inputs = params['numInputs']
        self.num_outputs = params['numOutputs']
        # self.R = params['r']
        if self.spiking:
            self.spikes = params['spikes']
            self.theta_0 = params['theta0']
            self.theta = params['theta']
            self.theta_last = params['thetaLast']
            self.m = params['m']
            self.tau_theta = params['tauTheta']
            self.g_max_spike = params['gMaxSpike']
            self.g_spike = params['gSpike']
            self.tau_syn = params['tauSyn']
            self.time_factor_threshold = params['timeFactorThreshold']
            self.time_factor_synapse = params['timeFactorSynapse']
            self.output_spike_connectivity = params['outConnSpike']
        if self.delay:
            self.spike_delays = params['spikeDelays']
            self.spike_rows = params['spikeRows']
            self.spike_cols = params['spikeCols']
            self.buffer_steps = params['bufferSteps']
            self.buffer_nrns = params['bufferNrns']
            self.delayed_spikes = params['delayedSpikes']
            self.spike_buffer = params['spikeBuffer']
        if self.electrical:
            self.g_electrical = params['gElectrical']
        if self.electrical_rectified:
            self.g_rectified = params['gRectified']
        if self.gated:
            self.g_ion = params['gIon']
            self.e_ion = params['eIon']
            self.pow_a = params['powA']
            self.slope_a = params['slopeA']
            self.k_a = params['kA']
            self.e_a = params['eA']
            self.pow_b = params['powB']
            self.slope_b = params['slopeB']
            self.k_b = params['kB']
            self.e_b = params['eB']
            self.tau_max_b = params['tauMaxB']
            self.pow_c = params['powC']
            self.slope_c = params['slopeC']
            self.k_c = params['kC']
            self.e_c = params['eC']
            self.tau_max_c = params['tauMaxC']
            self.b_gate = params['bGate']
            self.b_gate_last = params['bGateLast']
            self.b_gate_0 = params['bGate0']
            self.c_gate = params['cGate']
            self.c_gate_last = params['cGateLast']
            self.c_gate_0 = params['cGate0']

    def __call__(self, x=None):
        return self.forward(x)

    def reset(self):
        raise NotImplementedError

class SNS_Numpy(Backend):
    def __init__(self, params: Dict) -> None:
        super().__init__(params)

    def forward(self, x=None):
        self.V_last = np.copy(self.V)
        if x is None:
            i_app = 0
        else:
            i_app = np.matmul(self.input_connectivity, x)  # Apply external current sources to their destinations
        g_syn = np.maximum(0, np.minimum(self.g_max_non * ((self.V_last - self.e_lo) / (self.e_hi - self.e_lo)), self.g_max_non))
        if self.spiking:
            self.theta_last = np.copy(self.theta)
            self.g_spike = self.g_spike * (1 - self.time_factor_synapse)
            g_syn += self.g_spike

        i_syn = np.sum(g_syn * self.del_e, axis=1) - self.V_last * np.sum(g_syn, axis=1)
        if self.electrical:
            i_syn += (np.sum(self.g_electrical * self.V_last, axis=1) - self.V_last * np.sum(self.g_electrical, axis=1))
        if self.electrical_rectified:
            # create mask
            mask = np.subtract.outer(self.V_last, self.V_last).transpose() > 0
            masked_g = mask * self.g_rectified
            diag_masked = masked_g + masked_g.transpose() - np.diag(masked_g.diagonal())
            i_syn += np.sum(diag_masked * self.V_last, axis=1) - self.V_last * np.sum(diag_masked, axis=1)
        if self.gated:
            a_inf = 1 / (1 + self.k_a * np.exp(self.slope_a * (self.e_a - self.V_last)))
            b_inf = 1 / (1 + self.k_b * np.exp(self.slope_b * (self.e_b - self.V_last)))
            c_inf = 1 / (1 + self.k_c * np.exp(self.slope_c * (self.e_c - self.V_last)))

            tau_b = self.tau_max_b * b_inf * np.sqrt(self.k_b * np.exp(self.slope_b * (self.e_b - self.V_last)))
            tau_c = self.tau_max_c * c_inf * np.sqrt(self.k_c * np.exp(self.slope_c * (self.e_c - self.V_last)))

            self.b_gate_last = np.copy(self.b_gate)
            self.c_gate_last = np.copy(self.c_gate)

            self.b_gate = self.b_gate_last + self.dt * ((b_inf - self.b_gate_last) / tau_b)
            self.c_gate = self.c_gate_last + self.dt * ((c_inf - self.c_gate_last) / tau_c)

            i_ion = self.g_ion * (a_inf ** self.pow_a) * (self.b_gate ** self.pow_b) * (self.c_gate ** self.pow_c) * (
                        self.e_ion - self.V_last)
            i_gated = np.sum(i_ion, axis=0)

            self.V = self.V_last + self.time_factor_membrane * (
                        -self.g_m * (self.V_last - self.V_rest) + self.i_b + i_syn + i_app + i_gated)  # Update membrane potential
        else:
            self.V = self.V_last + self.time_factor_membrane * (
                        -self.g_m * (self.V_last - self.V_rest) + self.i_b + i_syn + i_app)  # Update membrane potential
        if self.spiking:
            self.theta = self.theta_last + self.time_factor_threshold * (
                        -self.theta_last + self.theta_0 + self.m * (self.V_last - self.V_rest))  # Update the firing thresholds
            self.spikes = np.sign(np.minimum(0, self.theta - self.V))  # Compute which neurons have spiked

            # New stuff with delay
            if self.delay:
                self.spike_buffer = np.roll(self.spike_buffer, 1, axis=0)  # Shift buffer entries down
                self.spike_buffer[0, :] = self.spikes  # Replace row 0 with the current spike data
                # Update a matrix with all of the appropriately delayed spike values
                self.delayed_spikes[self.spike_rows, self.spike_cols] = self.spike_buffer[
                    self.buffer_steps, self.buffer_nrns]

                self.g_spike = np.maximum(self.g_spike, (
                    -self.delayed_spikes) * self.g_max_spike)  # Update the conductance of connections which spiked
            else:
                self.g_spike = np.maximum(self.g_spike, (
                    -self.spikes) * self.g_max_spike)  # Update the conductance of connections which spiked
            self.V = ((self.V-self.V_rest) * (self.spikes + 1))+self.V_rest  # Reset the membrane voltages of neurons which spiked
        self.outputs = np.matmul(self.output_voltage_connectivity, self.V)
        if self.spiking:
            self.outputs += np.matmul(self.output_spike_connectivity, -self.spikes)

        return self.outputs

    def reset(self):
        self.V = np.copy(self.V_0)
        self.V_last = np.copy(self.V_0)
        if self.spiking:
            self.theta = np.copy(self.theta_0)
            self.theta_last = np.copy(self.theta_0)
        if self.gated:
            self.b_gate = np.copy(self.b_gate_0)
            self.b_gate_last = np.copy(self.b_gate_0)
            self.c_gate = np.copy(self.c_gate_0)
            self.c_gate_last = np.copy(self.c_gate_0)

class SNS_Torch(Backend):
    def __init__(self, params: Dict) -> None:
        super().__init__(params)

    def forward(self, x=None):
        self.V_last = torch.clone(self.V)
        if x is None:
            i_app = 0
        else:
            i_app = torch.matmul(self.input_connectivity, x)  # Apply external current sources to their destinations
        g_syn = torch.clamp(torch.minimum(self.g_max_non * ((self.V_last - self.e_lo) / (self.e_hi - self.e_lo)), self.g_max_non),min=0)
        if self.spiking:
            self.theta_last = torch.clone(self.theta)
            self.g_spike = self.g_spike * (1 - self.time_factor_synapse)
            g_syn += self.g_spike
        i_syn = torch.sum(g_syn * self.del_e, 1) - self.V_last * torch.sum(g_syn, 1)
        if self.electrical:
            i_syn += (torch.sum(self.g_electrical * self.V_last, 1) - self.V_last * torch.sum(self.g_electrical, 1))
        if self.electrical_rectified:
            # create mask
            mask = (self.V_last.reshape(-1,1)-self.V_last).transpose(0,1) > 0
            masked_g = mask * self.g_rectified
            diag_masked = masked_g + masked_g.transpose(0,1) - torch.diag(masked_g.diagonal())
            i_syn += torch.sum(diag_masked * self.V_last, 1) - self.V_last * torch.sum(diag_masked, 1)
        if self.gated:
            a_inf = 1 / (1 + self.k_a * torch.exp(self.slope_a*(self.e_a-self.V_last)))
            b_inf = 1 / (1 + self.k_b * torch.exp(self.slope_b*(self.e_b-self.V_last)))
            c_inf = 1 / (1 + self.k_c * torch.exp(self.slope_c*(self.e_c-self.V_last)))

            tau_b = self.tau_max_b * b_inf * torch.sqrt(self.k_b*torch.exp(self.slope_b*(self.e_b-self.V_last)))
            tau_c = self.tau_max_c * c_inf * torch.sqrt(self.k_c*torch.exp(self.slope_c*(self.e_c-self.V_last)))

            self.b_gate_last = torch.clone(self.b_gate)
            self.c_gate_last = torch.clone(self.c_gate)

            self.b_gate = self.b_gate_last + self.dt * ((b_inf - self.b_gate_last) / tau_b)
            self.c_gate = self.c_gate_last + self.dt * ((c_inf - self.c_gate_last) / tau_c)

            i_ion = self.g_ion*(a_inf**self.pow_a)*(self.b_gate**self.pow_b)*(self.c_gate**self.pow_c)*(self.e_ion-self.V_last)
            i_gated = torch.sum(i_ion, 0)

            self.V = self.V_last + self.time_factor_membrane * (-self.g_m * (self.V_last - self.V_rest) + self.i_b + i_syn + i_app + i_gated)  # Update membrane potential
        else:
            self.V = self.V_last + self.time_factor_membrane * (-self.g_m * (self.V_last - self.V_rest) + self.i_b + i_syn + i_app)  # Update membrane potential
        if self.spiking:
            self.theta = self.theta_last + self.time_factor_threshold * (-self.theta_last + self.theta_0 + self.m * (self.V_last - self.V_rest))  # Update the firing thresholds
            self.spikes = torch.sign(torch.clamp(self.theta - self.V,max=0))  # Compute which neurons have spiked

            # New stuff with delay
            if self.delay:
                self.spike_buffer = torch.roll(self.spike_buffer, 1, 0)   # Shift buffer entries down
                self.spike_buffer[0, :] = self.spikes    # Replace row 0 with the current spike data
                # Update a matrix with all of the appropriately delayed spike values
                self.delayed_spikes[self.spike_rows, self.spike_cols] = self.spike_buffer[self.buffer_steps, self.buffer_nrns]

                self.g_spike = torch.maximum(self.g_spike, (-self.delayed_spikes) * self.g_max_spike)  # Update the conductance of connections which spiked
            else:
                self.g_spike = torch.maximum(self.g_spike, (-self.spikes) * self.g_max_spike)  # Update the conductance of connections which spiked
            self.V = ((self.V - self.V_rest) * (self.spikes + 1)) + self.V_rest  # Reset the membrane voltages of neurons which spiked
        self.outputs = torch.matmul(self.output_voltage_connectivity, self.V)
        if self.spiking:
            self.outputs += torch.matmul(self.output_spike_connectivity, -self.spikes)

        return self.outputs

    def reset(self):
        self.V = torch.clone(self.V_0)
        self.V_last = torch.clone(self.V_0)
        if self.spiking:
            self.theta = torch.clone(self.theta_0)
            self.theta_last = torch.clone(self.theta_0)
        if self.gated:
            self.b_gate = torch.clone(self.b_gate_0)
            self.b_gate_last = torch.clone(self.b_gate_0)
            self.c_gate = torch.clone(self.c_gate_0)
            self.c_gate_last = torch.clone(self.c_gate_0)

class SNS_Sparse(Backend):
    def __init__(self, params: Dict) -> None:
        super().__init__(params)

    def forward(self, x=None):
        self.V_last = torch.clone(self.V)

        if x is None:
            i_app = 0
        else:
            i_app = torch.matmul(self.input_connectivity, x)  # Apply external current sources to their destinations
            i_app = i_app.to_sparse()

        g_syn = torch.clamp(torch.minimum(self.g_max_non.to_dense() * ((self.V_last - self.e_lo.to_dense()) / (self.e_hi - self.e_lo.to_dense())), self.g_max_non.to_dense()),
                            min=0)
        g_syn = g_syn.to_sparse()

        if self.spiking:
            self.theta_last = torch.clone(self.theta)
            self.g_spike = self.g_spike.to_dense() * (1 - self.time_factor_synapse)
            self.g_spike = self.g_spike.to_sparse()

            g_syn += self.g_spike

        if g_syn._nnz() > 0:
            i_syn = torch.sparse.sum(g_syn * self.del_e, 1) - (self.V_last * torch.sum(g_syn.to_dense(), 1)).to_sparse()
        else:
            i_syn = torch.sparse.sum(g_syn * self.del_e) - self.V_last * torch.sparse.sum(g_syn)
        if self.electrical:
            i_syn += (torch.sum(self.g_electrical.to_dense() * self.V_last, 1).to_sparse() -
                      (self.V_last * torch.sum(self.g_electrical.to_dense(), 1)).to_sparse())
        if self.electrical_rectified:
            # create mask
            mask = (self.V_last.reshape(-1, 1) - self.V_last).transpose(0, 1) > 0
            masked_g = mask * self.g_rectified.to_dense()
            diag_masked = masked_g + masked_g.transpose(0, 1) - torch.diag(masked_g.diagonal())
            i_syn += torch.sum(diag_masked * self.V_last, 1).to_sparse() - (
                        self.V_last * torch.sum(diag_masked, 1)).to_sparse()
        if self.gated:
            a_inf = (1 / (1 + self.k_a * torch.exp(
                self.slope_a.to_dense() * (self.e_a.to_dense() - self.V_last)))).to_sparse()
            b_inf = (1 / (1 + self.k_b * torch.exp(
                self.slope_b.to_dense() * (self.e_b.to_dense() - self.V_last)))).to_sparse()
            c_inf = (1 / (1 + self.k_c * torch.exp(
                self.slope_c.to_dense() * (self.e_c.to_dense() - self.V_last)))).to_sparse()

            tau_b = (self.tau_max_b * b_inf.to_dense() * torch.sqrt(
                self.k_b * torch.exp(self.slope_b.to_dense() * (self.e_b.to_dense() - self.V_last)))).to_sparse()
            tau_c = (self.tau_max_c * c_inf.to_dense() * torch.sqrt(
                self.k_c * torch.exp(self.slope_c.to_dense() * (self.e_c.to_dense() - self.V_last)))).to_sparse()

            self.b_gate_last = torch.clone(self.b_gate)
            self.c_gate_last = torch.clone(self.c_gate)

            self.b_gate = (self.b_gate_last.to_dense() + self.dt * (
                        (b_inf - self.b_gate_last).to_dense() / tau_b.to_dense())).to_sparse()
            self.c_gate = (self.c_gate_last.to_dense() + self.dt * (
                        (c_inf - self.c_gate_last).to_dense() / tau_c.to_dense())).to_sparse()

            i_ion = (self.g_ion.to_dense() * (a_inf.to_dense() ** self.pow_a.to_dense()) * (
                        self.b_gate.to_dense() ** self.pow_b.to_dense()) * (
                                 self.c_gate.to_dense() ** self.pow_c.to_dense()) * (
                                 self.e_ion.to_dense() - self.V_last)).to_sparse()
            i_gated = torch.sum(i_ion.to_dense(), 0).to_sparse()

            self.V = self.V_last + self.time_factor_membrane * (-self.g_m * (self.V_last - self.V_rest) + (self.i_b.to_dense())[0,
                                                                                          :] + i_syn + i_app + i_gated)  # Update membrane potential
        else:
            self.V = self.V_last + self.time_factor_membrane * (-self.g_m * (self.V_last - self.V_rest) + (self.i_b.to_dense())[0,
                                                                                          :] + i_syn + i_app)  # Update membrane potential
        if self.spiking:
            self.theta = self.theta_last + self.time_factor_threshold * (
                        -self.theta_last + self.theta_0 + (self.m.to_dense())[0,
                                                          :] * (self.V_last - self.V_rest))  # Update the firing thresholds

            self.spikes = torch.sign(torch.clamp(self.theta - self.V, max=0))  # Compute which neurons have spiked
            self.spikes = self.spikes.to_sparse()

            if self.delay:
                # New stuff with delay
                self.spike_buffer = self.spike_buffer.to_dense()
                self.spike_buffer = torch.roll(self.spike_buffer, 1, 0)  # Shift buffer entries down
                self.spike_buffer[0, :] = self.spikes.to_dense()  # Replace row 0 with the current spike data
                self.spike_buffer = self.spike_buffer.to_sparse()

                # Update a matrix with all of the appropriately delayed spike values
                self.delayed_spikes = self.delayed_spikes.to_dense()
                self.delayed_spikes[self.spike_rows, self.spike_cols] = (self.spike_buffer.to_dense())[
                    self.buffer_steps, self.buffer_nrns]
                self.delayed_spikes = self.delayed_spikes.to_sparse()

                self.g_spike = torch.maximum(self.g_spike.to_dense(), ((
                                                                           -self.delayed_spikes) * self.g_max_spike).to_dense())  # Update the conductance of connections which spiked
            else:
                self.g_spike = torch.maximum(self.g_spike.to_dense(), (
                    -self.spikes.to_dense()) * self.g_max_spike.to_dense())  # Update the conductance of connections which spiked
            self.g_spike = self.g_spike.to_sparse()
            self.V = ((self.V - self.V_rest) * (self.spikes.to_dense() + 1)) + self.V_rest  # Reset the membrane voltages of neurons which spiked
        self.outputs = torch.matmul(self.output_voltage_connectivity, self.V)
        if self.spiking:
            self.outputs += torch.matmul(self.output_spike_connectivity, -self.spikes.to_dense())

        return self.outputs

    def reset(self):
        self.V = torch.clone(self.V_0)
        self.V_last = torch.clone(self.V_0)
        if self.spiking:
            self.theta = torch.clone(self.theta_0)
            self.theta_last = torch.clone(self.theta_0)
        if self.gated:
            self.b_gate = torch.clone(self.b_gate_0)
            self.b_gate_last = torch.clone(self.b_gate_0)
            self.c_gate = torch.clone(self.c_gate_0)
            self.c_gate_last = torch.clone(self.c_gate_0)

class SNS_Iterative(Backend):
    def __init__(self, params: Dict) -> None:
        super().__init__(params)

    def set_params(self, params: Dict) -> None:
        self.dt = params['dt']
        self.name = params['name']
        self.spiking = params['spiking']
        self.delay = params['delay']
        self.electrical = params['elec']
        self.electrical_rectified = params['rect']
        self.gated = params['gated']
        self.num_channels = params['numChannels']
        self.V = params['v']
        self.V_last = params['vLast']
        self.V_0 = params['v0']
        self.V_rest = params['vRest']
        self.c_m = params['cM']
        self.g_m = params['gM']
        self.i_b = params['iB']
        self.time_factor_membrane = params['timeFactorMembrane']
        self.input_connectivity = params['inputConn']
        self.output_voltage_connectivity = params['outConnVolt']
        self.num_populations = params['numPop']
        self.num_neurons = params['numNeurons']
        self.num_connections = params['numConn']
        self.num_inputs = params['numInputs']
        self.num_outputs = params['numOutputs']
        # self.R = params['r']
        self.incoming_synapses = params['incomingSynapses']
        if self.spiking:
            self.spikes = params['spikes']
            self.theta_0 = params['theta0']
            self.theta = params['theta']
            self.theta_last = params['thetaLast']
            self.m = params['m']
            self.tau_theta = params['tauTheta']
            self.time_factor_threshold = params['timeFactorThreshold']
            self.output_spike_connectivity = params['outConnSpike']
        if self.delay:
            foo = 5
        if self.electrical:
            foo = 5
        if self.electrical_rectified:
            foo = 5
        if self.gated:
            self.g_ion = params['gIon']
            self.e_ion = params['eIon']
            self.pow_a = params['powA']
            self.slope_a = params['slopeA']
            self.k_a = params['kA']
            self.e_a = params['eA']
            self.pow_b = params['powB']
            self.slope_b = params['slopeB']
            self.k_b = params['kB']
            self.e_b = params['eB']
            self.tau_max_b = params['tauMaxB']
            self.pow_c = params['powC']
            self.slope_c = params['slopeC']
            self.k_c = params['kC']
            self.e_c = params['eC']
            self.tau_max_c = params['tauMaxC']
            self.b_gate = params['bGate']
            self.b_gate_last = params['bGateLast']
            self.b_gate_0 = params['bGate0']
            self.c_gate = params['cGate']
            self.c_gate_last = params['cGateLast']
            self.c_gate_0 = params['cGate0']

    def forward(self, x=None):
        self.V_last = np.copy(self.V)
        if self.spiking:
            self.theta_last = np.copy(self.theta)
        if self.gated:
            self.b_gate_last = np.copy(self.b_gate)
            self.c_gate_last = np.copy(self.c_gate)

        if x is None:
            i_app = np.zeros(self.num_neurons)
        else:
            i_app = np.matmul(self.input_connectivity, x)  # Apply external current sources to their destinations

        for nrn in range(self.num_neurons):
            i_syn = 0
            for syn in range(len(self.incoming_synapses[nrn])):
                neuron_src = self.incoming_synapses[nrn][syn]
                if neuron_src[1]:  # if spiking
                    neuron_src[5] = neuron_src[5] * (1 - neuron_src[6])
                    i_syn += neuron_src[5] * (neuron_src[4] - self.V_last[nrn])
                elif neuron_src[2]:  # if electrical
                    if neuron_src[4]:  # if rectified
                        if self.V_last[neuron_src[5]] > self.V_last[neuron_src[6]]:
                            i_syn += neuron_src[3] * (self.V_last[neuron_src[0]] - self.V_last[nrn])
                    else:
                        i_syn += neuron_src[3] * (self.V_last[neuron_src[0]] - self.V_last[nrn])
                else:  # if chemical
                    neuron_src[5] = np.maximum(0, np.minimum(neuron_src[3] * ((self.V_last[neuron_src[0]] - neuron_src[6]) / (neuron_src[7] - neuron_src[6])),
                                                             neuron_src[3]))
                    i_syn += neuron_src[5] * (neuron_src[4] - self.V_last[nrn])
            i_gated = 0
            if self.gated:
                a_inf = 1 / (
                            1 + self.k_a[:, nrn] * np.exp(self.slope_a[:, nrn] * (self.e_a[:, nrn] - self.V_last[nrn])))
                b_inf = 1 / (
                            1 + self.k_b[:, nrn] * np.exp(self.slope_b[:, nrn] * (self.e_b[:, nrn] - self.V_last[nrn])))
                c_inf = 1 / (
                            1 + self.k_c[:, nrn] * np.exp(self.slope_c[:, nrn] * (self.e_c[:, nrn] - self.V_last[nrn])))

                tau_b = self.tau_max_b[:, nrn] * b_inf * np.sqrt(
                    self.k_b[:, nrn] * np.exp(self.slope_b[:, nrn] * (self.e_b[:, nrn] - self.V_last[nrn])))
                tau_c = self.tau_max_c[:, nrn] * c_inf * np.sqrt(
                    self.k_c[:, nrn] * np.exp(self.slope_c[:, nrn] * (self.e_c[:, nrn] - self.V_last[nrn])))

                self.b_gate[:, nrn] = self.b_gate_last[:, nrn] + self.dt * ((b_inf - self.b_gate_last[:, nrn]) / tau_b)
                self.c_gate[:, nrn] = self.c_gate_last[:, nrn] + self.dt * ((c_inf - self.c_gate_last[:, nrn]) / tau_c)

                i_ion = self.g_ion[:, nrn] * (a_inf ** self.pow_a[:, nrn]) * (
                            self.b_gate[:, nrn] ** self.pow_b[:, nrn]) * (self.c_gate[:, nrn] ** self.pow_c[:, nrn]) * (
                                    self.e_ion[:, nrn] - self.V_last[nrn])
                i_gated = np.sum(i_ion)

            self.V[nrn] = self.V_last[nrn] + self.time_factor_membrane[nrn] * (
                        -self.g_m[nrn] * (self.V_last[nrn] - self.V_rest[nrn]) + self.i_b[nrn] + i_syn + i_app[
                    nrn] + i_gated)  # Update membrane potential
            if self.spiking:
                # if self.theta_0[nrn] != sys.float_info.max:
                self.theta[nrn] = self.theta_last[nrn] + self.time_factor_threshold[nrn] * (
                            -self.theta_last[nrn] + self.theta_0[nrn] + self.m[nrn] * (self.V_last[nrn] - self.V_rest[nrn]))  # Update the firing thresholds
                self.spikes[nrn] = np.sign(
                    np.minimum(0, self.theta[nrn] - self.V[nrn]))  # Compute which neurons have spiked
        if self.spiking:
            for nrn in range(self.num_neurons):
                if self.delay:
                    # New stuff with delay
                    for syn in range(len(self.incoming_synapses[nrn])):
                        neuron_src = self.incoming_synapses[nrn][syn]
                        if neuron_src[1]:  # if spiking
                            neuron_src[7] = np.roll(neuron_src[7], 1)  # Shift buffer entries down
                            neuron_src[7][0] = self.spikes[neuron_src[0]]  # Replace row 0 with the current spike data
                            neuron_src[5] = np.maximum(neuron_src[5], (-neuron_src[7][-1]) * neuron_src[
                                3])  # Update the conductance of connections which spiked
                self.V[nrn] = ((self.V[nrn] - self.V_rest[nrn]) * (self.spikes[nrn] + 1)) + self.V_rest[nrn]  # Reset the membrane voltages of neurons which spiked
        self.outputs = np.matmul(self.output_voltage_connectivity, self.V)
        if self.spiking:
            self.outputs += np.matmul(self.output_spike_connectivity, -self.spikes)

        return self.outputs

    def reset(self):
        self.V = np.copy(self.V_0)
        self.V_last = np.copy(self.V_0)
        if self.spiking:
            self.theta = np.copy(self.theta_0)
            self.theta_last = np.copy(self.theta_0)
        if self.gated:
            self.b_gate = np.copy(self.b_gate_0)
            self.b_gate_last = np.copy(self.b_gate_0)
            self.c_gate = np.copy(self.c_gate_0)
            self.c_gate_last = np.copy(self.c_gate_0)
