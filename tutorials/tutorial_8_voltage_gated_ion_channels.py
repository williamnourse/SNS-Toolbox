"""
Tutorial demonstrating the use of neurons with voltage-gated ion channels.
"""

from sns_toolbox.design.neurons import NonSpikingNeuronWithGatedChannels, NonSpikingNeuronWithPersistentSodiumChannel
from sns_toolbox.design.connections import NonSpikingSynapse
from sns_toolbox.design.networks import Network

from sns_toolbox.simulate.backends import SNS_Numpy, SNS_Torch

import numpy as np
import torch
import matplotlib.pyplot as plt

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Gating functions
"""
def zinf(U, Kz, Sz, Ez):
    return 1/(1+Kz*np.exp(Sz*(Ez-U)))

def tauz(U, tauzmax, Kz, Sz, Ez):
    return tauzmax*zinf(U, Kz, Sz, Ez)*np.sqrt(Kz*np.exp(Sz*(Ez-U)))


U = np.linspace(0,20)

Kh = 0.5
Sh = -0.046
Eh = 0
tauhmax = 300

Km = 1
Sm = 0.046
Em = 20

plt.figure()
plt.subplot(1,2,1)
plt.plot(U,zinf(U, Kh, Sh, Eh),label='h')
plt.plot(U,zinf(U, Km, Sm, Em),label='m')
plt.legend()
plt.title('Zinf')
plt.xlabel('U (mV)')
plt.subplot(1,2,2)
plt.plot(U, tauz(U, tauhmax, Kh, Sh, Eh), label='h')
plt.xlabel('U (mV)')
plt.ylabel('Tau (ms)')
plt.title('TauZ')

"""
Single neuron
"""

Cm = 5
Gm = 1
Ena = 50
Er = -60
R = 20

S = 0.05
delEna = Ena - Er
Km = 1
Kh = 0.5
Em = -40
Eh = -60
delEm = Em-Er
delEh = Eh-Er
tauHmax = 300

Gna = Gm*R/(zinf(R, Km, S, delEm)*zinf(R, Kh, -S, delEh)*(delEna-R))

g_ion = torch.tensor([Gna])
e_ion = torch.tensor([delEna])

pow_a = torch.tensor([1])
k_a = torch.tensor([Km])
slope_a = torch.tensor([S])
e_a = torch.tensor([delEm])

pow_b = torch.tensor([1])
k_b = torch.tensor([Kh])
slope_b = torch.tensor([-S])
e_b = torch.tensor([delEh])
tau_max_b = torch.tensor([tauHmax])

pow_c = torch.tensor([0])
k_c = torch.tensor([1])
slope_c = torch.tensor([0])
e_c = torch.tensor([0])
tau_max_c = torch.tensor([1])

neuron_gated = NonSpikingNeuronWithGatedChannels(membrane_capacitance=Cm, membrane_conductance=Gm,g_ion=g_ion,e_ion=e_ion,
                                                 pow_a=pow_a,k_a=k_a,slope_a=slope_a,e_a=e_a,
                                                 pow_b=pow_b,k_b=k_b,slope_b=slope_b,e_b=e_b,tau_max_b=tau_max_b,
                                                 pow_c=pow_c,k_c=k_c,slope_c=slope_c,e_c=e_c,tau_max_c=tau_max_c,
                                                 name='HC',color='orange')

net = Network()
net.add_neuron(neuron_gated)
net.add_input(0)
net.add_output(0)

I = 0
tStart = 1000
tEnd = 4000

dt = 1
tMax = 5000

t = np.arange(0,tMax,dt)
numSteps = np.size(t)

Iapp = torch.zeros([numSteps,1])
Iapp[tStart:tEnd,:] = I

Ipert = torch.zeros([numSteps,1])
Ipert[1,0] = 1

model = SNS_Torch(net,dt=dt,device='cpu')
data = torch.zeros([len(t), net.get_num_outputs_actual()])
inputs = Iapp + Ipert

for i in range(len(t)):
    data[i] = model.forward(inputs[i])
data = data.transpose(0,1)

plt.figure()
plt.plot(t,data[:][0])
plt.xlabel('t (ms)')
plt.ylabel('U (mV)')
plt.title('Single Neuron')

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
2 Neuron Oscillator
"""


def minf(U):
    return zinf(U, Km, S, delEm)


def hinf(U):
    return zinf(U, Kh, -S, delEh)


def cpg(delta=-0.01):
    Cm = 5
    Gm = 1
    Ena = 50
    Er = -60
    R = 20

    S = 0.05
    delEna = Ena - Er
    Km = 1
    Kh = 0.5
    Em = -40
    Eh = -60
    delEm = Em-Er
    delEh = Eh-Er
    tauHmax = 300

    Gna = Gm*R/(zinf(R, Km, S, delEm)*zinf(R, Kh, -S, delEh)*(delEna-R))

    g_ion = torch.tensor([Gna])
    e_ion = torch.tensor([delEna])

    pow_m = torch.tensor([1])
    k_m = torch.tensor([Km])
    slope_m = torch.tensor([S])
    e_m = torch.tensor([delEm])

    pow_h = torch.tensor([1])
    k_h = torch.tensor([Kh])
    slope_h = torch.tensor([-S])
    e_h = torch.tensor([delEh])
    tau_max_h = torch.tensor([tauHmax])

    neuron_cpg = NonSpikingNeuronWithPersistentSodiumChannel(membrane_capacitance=Cm, membrane_conductance=Gm,
                                                             g_ion=g_ion,e_ion=e_ion,
                                                             pow_m=pow_m,k_m=k_m,slope_m=slope_m,e_m=e_m,
                                                             pow_h=pow_h,k_h=k_h,slope_h=slope_h,e_h=e_h,tau_max_h=tau_max_h,
                                                             name='HC',color='orange')

    Ein = -100
    delEsyn = Ein - R

    gSyn = (-delta - delta * Gna * minf(delta) * hinf(delta) + Gna * minf(delta) * hinf(delta) * delEna) / (
                delta - delEsyn)

    synapse_cpg = NonSpikingSynapse(max_conductance=gSyn, relative_reversal_potential=delEsyn)

    net = Network()
    net.add_neuron(neuron_cpg, name='HC0', color='blue')
    net.add_input('HC0')
    net.add_output('HC0')
    net.add_neuron(neuron_cpg, name='HC1', color='orange')
    net.add_output('HC1')
    net.add_connection(synapse_cpg, 'HC0', 'HC1')
    net.add_connection(synapse_cpg, 'HC1', 'HC0')

    I = 0
    tStart = 1000
    tEnd = 4000
    tTail = 2500

    dt = 1
    tMax = 5000

    t = np.arange(0, tMax, dt)
    numSteps = np.size(t)

    Iapp = torch.zeros([numSteps,1])
    Iapp[tStart:tEnd,:] = I

    Ipert = torch.zeros([numSteps,1])
    Ipert[1,0] = 1

    model = SNS_Torch(net, dt=dt, device='cpu')
    data = torch.zeros([len(t), net.get_num_outputs_actual()])
    inputs = Iapp + Ipert

    for i in range(len(t)):
        data[i] = model.forward(inputs[i])
    data = data.transpose(0,1)

    return data

def plot_cpg(delta=-0.01):
    data = cpg(delta=delta)
    plt.figure()
    plt.plot(t,data[:][0],label='HC0',color='C0')
    plt.plot(t,data[:][1],label='HC1',color='C1',linestyle='--')
    plt.xlabel('t (ms)')
    plt.ylabel('U (mV)')
    title = 'CPG: Delta = ' + str(delta)
    plt.title(title)
    plt.legend()


plot_cpg(delta=-0.01)
plot_cpg(delta=0.01)
plot_cpg(delta=0.1)

plt.show()