import os
import torch
from torch import nn
from torch.nn.parameter import Parameter
from typing import Dict, Optional, Type
from numbers import Number
from sns_toolbox.neurons import NonSpikingNeuron
from sns_toolbox.networks import Network
from sns_toolbox.connections import NonSpikingSynapse
import matplotlib.pyplot as plt
import numpy as np
import time

"""
Neurons
"""
class SNSModule(nn.Module):
    def __init__(self, dt: float, params: Dict, grads=None) -> None:
        super().__init__()
        if grads is None:
            grads = []
        self.net_params = {}
        self.params = nn.ParameterDict()
        for key, value in params.items():
            if isinstance(value, torch.Tensor):
                self.params[key] = Parameter(value, requires_grad=(key in grads))
            else:
                self.net_params[key] = value
        self.dt = dt

    def reset(self):
        raise NotImplementedError

class NonSpikingLayer(SNSModule):
    def __init__(self, dt: float, params: Dict, grads=None) -> None:
        super().__init__(dt, params, grads=grads)

        self.time_factor = self.dt / (self.params['cM']/self.params['gM'])
        self.v = torch.clone(self.params['v0'])
        self.v_last = torch.clone(self.params['v0'])

    def forward(self, x: Optional[torch.Tensor] = None):
        self.v_last = torch.clone(self.v)
        u = torch.sub(self.v_last, self.params['vRest']) # v_last - v_rest
        i_leak = torch.mul(-self.params['gM'],u) # -g_m*(v_last-v_rest)
        i_int = torch.add(i_leak, self.params['iB']) # -g_m*(v_last - v_rest) + Ib
        if x is None:
            i_sum = torch.clone(i_int)
        else:
            i_sum = torch.add(x, i_int) # -g_m*(v_last - v_rest) + Ib + I
        self.v = torch.addcmul(self.v_last, self.time_factor, i_sum)
        return self.v

    def reset(self):
        self.v = torch.clone(self.params['v0'])
        self.v_last = torch.clone(self.params['v0'])

    def set_to_rest(self):
        self.v = torch.clone(self.params['vRest'])
        self.v_last = torch.clone(self.params['vRest'])

class GatedLayer(NonSpikingLayer):
    def __init__(self, dt: float, params: Dict, grads=None) -> None:
        super().__init__(dt, params, grads=grads)

        self.b_gate_0 = 1 / (1 + self.params['kB'] * torch.exp(self.params['slopeB'] * (self.params['v0'] - self.params['eB'])))
        self.c_gate_0 = 1 / (1 + self.params['kC'] * torch.exp(self.params['slopeC'] * (self.params['v0'] - self.params['eC'])))

        self.b_gate = torch.clone(self.b_gate_0)
        self.b_gate_last = torch.clone(self.b_gate_0)
        self.c_gate = torch.clone(self.c_gate_0)
        self.c_gate_last = torch.clone(self.c_gate_0)

    def forward(self, x: Optional[torch.Tensor] = None):
        self.v_last = torch.clone(self.v)
        u = torch.sub(self.v_last, self.params['vRest'])  # v_last - v_rest
        i_leak = torch.mul(-self.params['gM'], u)  # -g_m*(v_last-v_rest)
        i_int = torch.add(i_leak, self.params['iB'])  # -g_m*(v_last - v_rest) + Ib
        if x is None:
            i_sum = torch.clone(i_int)
        else:
            i_sum = torch.add(x, i_int)  # -g_m*(v_last - v_rest) + Ib + I

        a_inf = 1 / (1 + self.params["kA"] * torch.exp(
            self.params["slopeA"] * (self.params["eA"] - self.v_last)))
        b_inf = 1 / (1 + self.params["kB"] * torch.exp(
            self.params["slopeB"] * (self.params["eB"] - self.v_last)))
        c_inf = 1 / (1 + self.params["kC"] * torch.exp(
            self.params["slopeC"] * (self.params["eC"] - self.v_last)))

        tau_b = self.params["tauMaxB"] * b_inf * torch.sqrt(
            self.params["kB"] * torch.exp(self.params["slopeB"] * (self.params["eB"] - self.v_last)))
        tau_c = self.params["tauMaxC"] * c_inf * torch.sqrt(
            self.params["kC"] * torch.exp(self.params["slopeC"] * (self.params["eC"] - self.v_last)))

        self.b_gate_last = torch.clone(self.b_gate)
        self.c_gate_last = torch.clone(self.c_gate)

        self.b_gate = self.b_gate_last + self.dt * ((b_inf - self.b_gate_last) / tau_b)
        self.c_gate = self.c_gate_last + self.dt * ((c_inf - self.c_gate_last) / tau_c)

        i_ion = self.params["gIon"] * (a_inf ** self.params["powA"]) * (self.b_gate ** self.params["powB"]) * (self.c_gate ** self.params["powC"]) * (self.params["eIon"] - self.v_last)
        i_gated = torch.sum(i_ion, 0)
        i_sum+=i_gated
        self.v = torch.addcmul(self.v_last, self.time_factor, i_sum)
        return self.v

    def reset(self):
        super().reset()
        self.b_gate = torch.clone(self.params['bGate0'])
        self.b_gate_last = torch.clone(self.params['bGate0'])
        self.c_gate = torch.clone(self.params['cGate0'])
        self.c_gate_last = torch.clone(self.params['cGate0'])

class SpikingUnit(nn.Module):
    def __init__(self, layer_type: Type[NonSpikingLayer], dt: float, params: Dict, grads=None) -> None:
        super().__init__()
        if grads is None:
            grads = []
        self.net_params = {}
        self.params = nn.ParameterDict()
        for key, value in params.items():
            if isinstance(value, torch.Tensor):
                self.params[key] = Parameter(value, requires_grad=(key in grads))
            else:
                self.net_params[key] = value
        self.layer = layer_type(dt, params, grads=grads)
        self.time_factor_threshold = dt / self.params['tauTheta']
        self.theta = torch.clone(self.params['theta0'])
        self.theta_last = torch.clone(self.params['theta0'])
        self.v = self.layer.v
        self.v_last = self.layer.v_last
        self.spikes = torch.zeros_like(self.v)

    def forward(self, x: Optional[torch.Tensor] = None):
        self.theta_last = torch.clone(self.theta)

        self.layer(x)

        # threshold
        theta_a = torch.addcmul(self.params['theta0'],self.params['m'],self.layer.v_last) # theta0 + m*v_last
        theta_b = torch.sub(theta_a,self.theta_last) # -theta_last + theta0 + m*v_last
        self.theta = torch.addcmul(self.theta_last,self.time_factor_threshold,theta_b) # theta_last + dt/tau*(-theta_last + theta0 + m*v_last)

        # spiking
        v_diff = torch.sub(self.theta,self.layer.v)
        clamped_diff = torch.clamp(v_diff,max=0)
        self.spikes = torch.sign(clamped_diff)

        # voltage_reset
        spikes_flipped = torch.add(self.spikes,1.0)
        v_diff = torch.sub(self.layer.v, self.layer.params['vRest'])
        self.layer.v = torch.addcmul(self.layer.params['vRest'],v_diff,spikes_flipped)

        self.v = self.layer.v

        return self.spikes

    def reset(self):
        self.layer.reset()
        self.theta = torch.clone(self.theta_0)
        self.theta_last = torch.clone(self.theta_0)



"""
Synapses
"""
class NonSpikingSynapseTransform(SNSModule):
    def __init__(self, dt: float, params: Dict, grads=None) -> None:
        super().__init__(dt, params, grads=grads)

    def forward(self,v_pre_last, v_post_last):
        r = torch.sub(self.params['eHi'],self.params['eLo']) # e_hi-e_lo
        offset_v = torch.sub(v_pre_last,self.params['eLo']) # v_last-e_lo
        scaled_v = torch.div(offset_v,r) # (v_last-e_lo)/(e_hi-e_lo)
        weighted_v = torch.mul(self.params['gMax'],scaled_v)
        g_syn = torch.clamp(weighted_v, min=torch.tensor(0.0), max=self.params['gMax'])

        g_rev = torch.mul(g_syn,self.params['reversal'])
        sum_g_rev = torch.sum(g_rev,1)
        sum_g = torch.sum(g_syn,1)
        v_g = torch.mul(v_post_last,sum_g)
        i_syn = torch.sub(sum_g_rev,v_g)
        return i_syn

    def reset(self):
        pass

class SpikingSynapseTransform(SNSModule):
    def __init__(self, dt: float, params: Dict, grads=None) -> None:
        super().__init__(dt, params, grads=grads)

        self.time_factor = self.dt / self.params['tauSyn']
        self.g_spike = torch.zeros_like(self.time_factor)

    def forward(self, spikes_pre_last, v_post_last):
        self.g_spike = torch.maximum(self.g_spike, (-spikes_pre_last)*self.params['gMax'])
        self.g_spike *= 1 - self.time_factor

        g_rev = torch.mul(self.g_spike, self.params['reversal'])
        sum_g_rev = torch.sum(g_rev, 1)
        sum_g = torch.sum(self.g_spike, 1)
        v_g = torch.mul(v_post_last, sum_g)
        i_syn = torch.sub(sum_g_rev, v_g)
        return i_syn

    def reset(self):
        self.g_spike = torch.zeros_like(self.time_factor)

class ElectricalSynapseTransform(SNSModule):
    def __init__(self, dt: float, params: Dict, grads=None) -> None:
        super().__init__(dt, params, grads=grads)

    def forward(self, v_pre_last, v_post_last):
        i_post = (torch.sum(self.params['gElectrical'] * v_pre_last, 1) - v_post_last * torch.sum(self.params['gElectrical'], 1))
        i_pre = (torch.sum(self.params['gElectrical'].transpose(0,1) * v_post_last, 1) - v_pre_last * torch.sum(self.params['gElectrical'].transpose(0,1), 1))
        return i_pre, i_post

    def reset(self):
        pass

class RectifiedElectricalSynapseTransform(SNSModule):
    def __init__(self, dt: float, params: Dict, grads=None) -> None:
        super().__init__(dt, params, grads=grads)

    def forward(self, v_pre_last, v_post_last):
        ones_matrix = torch.ones_like(self.params['gElectrical'])
        pre = ones_matrix * v_pre_last
        post = ones_matrix * v_post_last.reshape(-1,1)
        mask = (pre-post)>0
        masked_conductance = mask * self.params['gElectrical']
        i_post = (torch.sum(masked_conductance * v_pre_last, 1) - v_post_last * torch.sum(masked_conductance, 1))
        i_pre = (torch.sum(masked_conductance.transpose(0,1) * v_post_last, 1) - v_pre_last * torch.sum(masked_conductance.transpose(0,1), 1))
        return i_pre, i_post

    def reset(self):
        pass



# Torch version
jit = False
shape = [1]
dt = 0.1

v_0_torch = torch.zeros(shape)+0.0
v_rest_torch = torch.zeros(shape)+0.0
g_m_torch = torch.zeros(shape)+1.0
c_m_torch = torch.zeros(shape)+5.0
bias_torch = torch.zeros(shape)+0.0

g_ion = torch.zeros(shape)+1.0
e_ion = torch.zeros(shape)+10.0
pow_a = torch.tensor([1.0])
k_a = torch.tensor([1.0])
slope_a = torch.tensor([0.05])
e_a = torch.tensor([20.0])
pow_b = torch.tensor([1.0])
k_b = torch.tensor([0.5])
slope_b = torch.tensor([-0.05])
e_b = torch.tensor([0.0])
tau_max_b = torch.tensor([300.0])
pow_c = torch.tensor([0.0])
k_c = torch.tensor([1.0])
slope_c = torch.tensor([0.0])
e_c = torch.tensor([0.0])
tau_max_c = torch.tensor([1.0])


theta_0 = torch.ones(shape)
m = torch.zeros(shape) + 1.0
tau_theta = torch.zeros(shape)+5.0

neuron_params = {'v0': v_0_torch,
                 'vRest': v_rest_torch,
                 'gM': g_m_torch,
                 'cM': c_m_torch,
                 'iB': bias_torch,
                 'theta0': theta_0,
                 'm': m,
                 'tauTheta': tau_theta,
                 'gIon': g_ion,
                 'eIon': e_ion,
                 'powA': pow_a,
                 'kA': k_a,
                 'slopeA': slope_a,
                 'eA': e_a,
                 'powB': pow_b,
                 'kB': k_b,
                 'slopeB': slope_b,
                 'eB': e_b,
                 'tauMaxB': tau_max_b,
                 'powC': pow_c,
                 'kC': k_c,
                 'slopeC': slope_c,
                 'eC': e_c,
                 'tauMaxC': tau_max_c,}

# Synapse Params
g_max_torch = torch.zeros([1,1])+1.0
reversal_torch = torch.zeros([1,1])+5.0
threshold_low_torch = torch.zeros([1,1])+0.0
threshold_high_torch = torch.zeros([1,1])+1.0
tau_syn = torch.zeros([1,1])+5.0

synapse_params = {'gMax': g_max_torch,
                  'reversal': reversal_torch,
                  'eLo': threshold_low_torch,
                  'eHi': threshold_high_torch,
                  'tauSyn': tau_syn,
                  'gElectrical': g_max_torch}

class TorchNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_0 = NonSpikingLayer(dt, neuron_params)
        self.synapse = RectifiedElectricalSynapseTransform(dt, synapse_params)
        self.layer_1 = NonSpikingLayer(dt, neuron_params)
        self.i_pre = None
        self.i_post = None

    def forward(self, x: Optional[torch.Tensor] = None):
        if x is None:
            x = torch.tensor([0,0])
        self.i_pre, self.i_post = self.synapse(self.layer_0.v, self.layer_1.v)
        v_a = self.layer_0(x[0] + self.i_pre)
        v_b = self.layer_1(x[1] + self.i_post)
        return v_a, v_b



if jit:
    model_torch = torch.jit.script(TorchNet())
    # print(model_torch.code)
else:
    model_torch = TorchNet()



t = np.arange(0, 100, dt)
data_torch = torch.zeros([len(t),2])
data_syn = torch.zeros([len(t),2])
# model_torch = SpikingUnit(NonSpikingLayer, dt, neuron_params)
inp = torch.tensor([5.0, 0.0])

start = time.time()
print(len(t))
for i in range(len(t)):
    if (i > 500) and (i < 600):
        inp = torch.tensor([0.0, 5.0])
    else:
        inp = torch.tensor([5.0, 0.0])
    data_torch[i,:] = torch.tensor(model_torch(inp))
    data_syn[i,0] = model_torch.i_pre
    data_syn[i,1] = model_torch.i_post
end = time.time()
print('Total Torch Time: %f'%(end-start))
print('Avg Step Time: %f'%((end-start)/len(t)))

data_torch = data_torch.transpose(1,0)
data_syn = data_syn.transpose(1,0)

plt.figure()
plt.subplot(2,1,1)
plt.plot(t,data_torch[0,:].detach().numpy(), color='C0')
plt.plot(t,data_torch[1,:].detach().numpy(), color='C1')
plt.axvline(x=t[500],color='black')
plt.axvline(x=t[600],color='black')
plt.subplot(2,1,2)
plt.plot(t,data_syn[0,:].detach().numpy(), color='C0')
plt.plot(t,data_syn[1,:].detach().numpy(), color='C1')
plt.axvline(x=t[500],color='black')
plt.axvline(x=t[600],color='black')

plt.show()
