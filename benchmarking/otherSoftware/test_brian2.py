import numpy as np
from brian2 import *
import matplotlib.pyplot as plt

# print(20*volt)

start_scope()
num_neurons = 2
dt = 0.1
defaultclock.dt = dt*ms
tau = 10*ms
eqs_neuron = '''
dv/dt = (-v+I+Isyn)/tau : 1
I : 1
Isyn : 1
'''

pop = NeuronGroup(2,eqs_neuron, method='euler')
pop.I = [1,0]

eqs_synapse = '''
Isyn_post = Gmax*clip(v_pre/R, 0, 1)*(DelE - v_post) : 1 (summed)
Gmax : 1
R : 1
DelE : 1
'''
syn = Synapses(pop, pop, eqs_synapse)
syn.connect(i=0,j=1)
syn.Gmax = 0.5
syn.R = 1
syn.DelE = -3

tmax = 50 # ms
t = np.arange(0,tmax,dt)
data = np.zeros([len(t),num_neurons])
# G.v = 'rand()'
data[0,:] = pop.v
for i in range(1, len(data)):
    # print(G.v[0])
    print(i)
    run(0.1*ms)
    data[i,:] = pop.v
    # print(G.v[0])
data = data.transpose()
plt.figure()
for i in range(num_neurons):
    plt.plot(t,data[i,:])
plt.show()
