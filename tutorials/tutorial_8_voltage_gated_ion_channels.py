"""
Tutorial demonstrating the use of neurons with voltage-gated ion channels.
"""

from sns_toolbox.neurons import NonSpikingNeuronWithGatedChannels, NonSpikingNeuronWithPersistentSodiumChannel
from sns_toolbox.connections import NonSpikingSynapse
from sns_toolbox.networks import Network

import numpy as np
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

g_ion = np.array([Gna])
e_ion = np.array([delEna])

pow_a = np.array([1])
k_a = np.array([Km])
slope_a = np.array([S])
e_a = np.array([delEm])

pow_b = np.array([1])
k_b = np.array([Kh])
slope_b = np.array([-S])
e_b = np.array([delEh])
tau_max_b = np.array([tauHmax])

pow_c = np.array([0])
k_c = np.array([1])
slope_c = np.array([0])
e_c = np.array([0])
tau_max_c = np.array([1])

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

Iapp = np.zeros([numSteps,1])
Iapp[tStart:tEnd,:] = I

Ipert = np.zeros([numSteps,1])
Ipert[1,0] = 1

model = net.compile(backend='numpy',dt=dt)
data = np.zeros([len(t), net.get_num_outputs_actual()])
inputs = Iapp + Ipert

for i in range(len(t)):
    data[i] = model(inputs[i])
data = data.transpose()

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

    g_ion = np.array([Gna])
    e_ion = np.array([delEna])

    k_m = np.array([Km])
    slope_m = np.array([S])
    e_m = np.array([delEm])

    k_h = np.array([Kh])
    slope_h = np.array([-S])
    e_h = np.array([delEh])
    tau_max_h = np.array([tauHmax])

    neuron_cpg = NonSpikingNeuronWithPersistentSodiumChannel(membrane_capacitance=Cm, membrane_conductance=Gm,
                                                             g_ion=g_ion,e_ion=e_ion,
                                                             k_m=k_m,slope_m=slope_m,e_m=e_m,
                                                             k_h=k_h,slope_h=slope_h,e_h=e_h,tau_max_h=tau_max_h,
                                                             name='HC',color='orange')

    Ein = -100
    delEsyn = Ein - R

    gSyn = (-delta - delta * Gna * minf(delta) * hinf(delta) + Gna * minf(delta) * hinf(delta) * delEna) / (
                delta - delEsyn)

    synapse_cpg = NonSpikingSynapse(max_conductance=gSyn, reversal_potential=delEsyn)

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

    dt = 1
    tMax = 5000

    t = np.arange(0, tMax, dt)
    numSteps = np.size(t)

    Iapp = np.zeros([numSteps,1])
    Iapp[tStart:tEnd,:] = I

    Ipert = np.zeros([numSteps,1])
    Ipert[1,0] = 1

    model = net.compile(backend='numpy', dt=dt)
    data = np.zeros([len(t), net.get_num_outputs_actual()])
    inputs = Iapp + Ipert

    for i in range(len(t)):
        data[i] = model(inputs[i])
    data = data.transpose()

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