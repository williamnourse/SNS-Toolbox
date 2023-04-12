from sns_toolbox.networks import Network
from sns_toolbox.connections import NonSpikingSynapse, SpikingSynapse
from sns_toolbox.neurons import NonSpikingNeuron, SpikingNeuron

import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# SNS Version
net_ns = Network()
net_s = Network()
neuron_ns = NonSpikingNeuron(membrane_capacitance=5, membrane_conductance=1, resting_potential=0.0, bias=0.0)
neuron_s = SpikingNeuron(membrane_conductance=1, membrane_capacitance=5, threshold_proportionality_constant=1, threshold_initial_value=1,resting_potential=0, bias=0, threshold_time_constant=5.0)

conn_ns = NonSpikingSynapse(max_conductance=1.0, reversal_potential=5, e_lo=0, e_hi=1)
conn_s = SpikingSynapse(max_conductance=1.0, reversal_potential=5, time_constant=1, transmission_delay=0)

net_ns.add_neuron(neuron_ns,name='Source')
net_ns.add_neuron(neuron_ns,name='Dest')
net_ns.add_connection(conn_ns,'Source', 'Dest')

net_s.add_neuron(neuron_s,name='Source')
net_s.add_neuron(neuron_s,name='Dest')
net_s.add_connection(conn_s,'Source', 'Dest')

net_ns.add_input('Source')
net_s.add_input('Source')

net_ns.add_output('Source')
net_ns.add_output('Dest')
net_s.add_output('Source')
net_s.add_output('Dest')

dt = 0.01
backend = 'numpy'

model_ns = net_ns.compile(dt=dt, backend=backend)
model_s = net_s.compile(dt=dt, backend=backend)


# Evaluate
t = np.arange(0,50,dt)
data_ns = np.zeros([len(t),2])
data_s = np.zeros_like(data_ns)
times_ns = np.zeros_like(t)
times_s = np.zeros_like(t)
stim_ns = 2.0
sns_ns_start = time.time()
for i in tqdm(range(len(t))):
    t_ns_start = time.time()
    data_ns[i,:] = model_ns([stim_ns])
    t_ns_end = time.time()
    times_ns[i] = t_ns_end-t_ns_start
sns_ns_end = time.time()
print('SNS NS Total Time: %f'%(sns_ns_end-sns_ns_start))

stim_s = 5.0
sns_s_start = time.time()
g_sns = np.zeros_like(t)
for i in tqdm(range(len(t))):
    t_s_start = time.time()
    g_sns[i] = model_s.g_spike[1,0]
    data_s[i, :] = model_s([5])
    t_s_end = time.time()
    times_s[i] = t_s_end - t_s_start
sns_s_end = time.time()
print('SNS S Total Time: %f'%(sns_s_end-sns_s_start))
sim_time = t


from ANNarchy import *

setup(dt=0.01, paradigm="cuda")
NonSpikingNeuron = Neuron(
    parameters="""
        Cm = 5.0
        Gm = 1.0
        bias = 0.0
        Esyn = 5.0
    """,
    equations="""
        Cm * dv/dt = -Gm * v + bias + sum(exc)*(Esyn-v)
        r = v
    """
)
NonSpikingSynapse = Synapse(
    parameters="""
        Gmax = 1.0
        Esyn = 5.0
        El = 0.0
        Eh = 1.0
    """,
    equations="""
        w = clip(Gmax * (pre.r-El)/(Eh-El), 0.0, Gmax) 
    """,
    psp="""
        w
    """
)
SpikingNeuron = Neuron(
    parameters="""
        Cm = 5.0
        Gm = 1.0
        bias = 0.0
        tau = 5.0
        To = 1.0
        m = 1.0
        tau_exc = 1.0
        Esyn = 5.0
    """,
    equations="""
        Cm * dv/dt = -Gm * v + bias + g_exc * (Esyn-v) : init = 0.0
        tau * dT/dt = -T + To + m * v : init = 1.0
        tau_exc * dg_exc/dt = -g_exc
    """,
    spike = "v > T",
    reset = "v = 0"
)
SpikingSynapse = Synapse(
    parameters="""
        Gmax = 0.99
        Esyn = 5.0
    """,
    equations="""""",
    pre_spike="""
        g_target = Gmax : max = Gmax
    """
)
net_ns = Network()
pop0_ns = Population(geometry=1,neuron=NonSpikingNeuron)
pop1_ns = Population(geometry=1,neuron=NonSpikingNeuron)
proj_ns = Projection(
    pre=pop0_ns,
    post=pop1_ns,
    target='exc',
    synapse=NonSpikingSynapse
).connect_all_to_all(weights=0.0)
net_ns.add([pop0_ns,pop1_ns,proj_ns])

net_ns.compile()

ann_data_ns = np.zeros([len(sim_time),2])
ann_times_ns = np.zeros_like(sim_time)

ann_ns_start = time.time()
for i in tqdm(range(len(sim_time))):
    ann_t_ns_start = time.time()
    net_ns.get(pop0_ns).bias = stim_ns
    net_ns.step()
    ann_data_ns[i,:] = [net_ns.get(pop0_ns).r,net_ns.get(pop1_ns).r]
    ann_t_ns_end = time.time()
    ann_times_ns[i] = ann_t_ns_end-ann_t_ns_start
ann_ns_end = time.time()
print('ANN NS Total Time: %f'%(ann_ns_end-ann_ns_start))

net_s = Network()
pop0_s = Population(geometry=1,neuron=SpikingNeuron)
pop1_s = Population(geometry=1,neuron=SpikingNeuron)
proj_s = Projection(
    pre=pop0_s,
    post=pop1_s,
    target='exc',
    synapse=SpikingSynapse
).connect_one_to_one()
net_s.add([pop0_s,pop1_s,proj_s])

net_s.compile()

ann_data_s = np.zeros([len(sim_time),2])
ann_times_s = np.zeros_like(sim_time)

g_ann = np.zeros_like(sim_time)
ann_s_start = time.time()
for i in tqdm(range(len(sim_time))):
    ann_t_s_start = time.time()
    net_s.get(pop0_s).bias = stim_s
    g_ann[i] = net_s.get(pop1_s).g_exc
    net_s.step()
    ann_data_s[i,:] = [net_s.get(pop0_s).v,net_s.get(pop1_s).v]
    ann_t_s_end = time.time()
    ann_times_s[i] = ann_t_s_end-ann_t_s_start
ann_s_end = time.time()
print('ANN S Total Time: %f'%(ann_s_end-ann_s_start))

    # t_s_start = time.time()
    # data_s[i, :] = model_s([5])
    # t_s_end = time.time()
    # times_s[i] = t_s_end - t_s_start

def plot_data(t,data,times):
    plt.subplot(3,1,1)
    plt.plot(t,data.transpose()[0,:])
    plt.subplot(3,1,2)
    plt.plot(t,data.transpose()[1,:])
    plt.subplot(3,1,3)
    plt.plot(t,times)

print('Mean Spiking: %f'%np.mean(times_s))
print('Mean Nonspiking: %f'%np.mean(times_ns))
print('ANNarchy Mean Spiking: %f'%np.mean(ann_times_s))
print('ANNarchy Mean Nonspiking: %f'%np.mean(ann_times_ns))

plt.figure()
plot_data(t,data_ns,times_ns)
plot_data(t,ann_data_ns,ann_times_ns)

plt.figure()
plot_data(t, data_s, times_s)
plot_data(t,ann_data_s,ann_times_s)

plt.figure()
plt.plot(t,g_sns)
plt.plot(sim_time,g_ann)
plt.title('Cuda')

plt.show()
