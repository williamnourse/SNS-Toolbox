import numpy as np
from brian2 import *
import matplotlib.pyplot as plt

# print(20*volt)

start_scope()
defaultclock.dt = 0.1*ms
tau = 10*ms
eqs = '''
dv/dt = (5-v)/tau : 1
'''

G = NeuronGroup(5,eqs, method='euler')

tmax = 100 # ms
t = np.arange(0,tmax,0.1)
data = np.zeros([len(t),5])
G.v = 'rand()'
data[0,:] = G.v
for i in range(1, len(data)):
    # print(G.v[0])
    print(i)
    run(0.1*ms)
    data[i,:] = G.v
    # print(G.v[0])
data = data.transpose()
plt.figure()
plt.plot(t,data[0,:])
plt.plot(t,data[1,:])
plt.plot(t,data[2,:])
plt.plot(t,data[3,:])
plt.plot(t,data[4,:])
plt.show()