import sys
import numpy as np
import nengo
import time
import pickle
import matplotlib.pyplot as plt

from sns_toolbox.neurons import NonSpikingNeuron
from sns_toolbox.connections import NonSpikingSynapse
from sns_toolbox.networks import Network

class SpikingNeuronsWithSynapticInput(nengo.Process):
    def __init__(self, num_neurons, Cm, Gmax, DelE, tau, m, theta0=1, bias=None, R=1, **kwargs):
        self.num_neurons = num_neurons

        if bias is None:
            self.bias = np.zeros(num_neurons)
        elif len(bias) == 1:
            self.bias = np.zeros(num_neurons) + bias[0]
        else:
            self.bias = bias
        self.R = R
        self.Cm = Cm
        self.Gmax = Gmax
        self.DelE = DelE
        self.tau = tau
        self.m = m
        self.theta0 = theta0
        self.theta = np.zeros(num_neurons) + theta0
        self.spikes = np.zeros_like(self.theta)
        self.gsyn = np.zeros([self.num_neurons, self.num_neurons])

        super().__init__(default_size_in=num_neurons, default_size_out=num_neurons, **kwargs)

    def make_state(self, shape_in, shape_out, dt, dtype=None):
        u = np.zeros(self.num_neurons)

        return {'state': u}

    def make_step(self, shape_in, shape_out, dt, rng, state):
        bias = self.bias
        Cm = self.Cm
        R = self.R
        u = state['state']
        Gmax = self.Gmax
        DelE = self.DelE
        tau = self.tau
        m = self.m
        theta0 = self.theta0

        def step(t, x):
            self.gsyn *= 1 - dt*1000/tau
            # print(self.gsyn)
            i_syn = np.sum(self.gsyn * DelE, axis=1) - u * np.sum(self.gsyn, axis=1)
            # print(i_syn)
            u[:] += dt*1000/Cm*(-u + i_syn + bias)
            # print(u)
            self.theta += dt*1000/tau*(-self.theta + theta0 + m*u)
            self.spikes = np.sign(np.minimum(0, self.theta - u))
            self.gsyn = np.maximum(self.gsyn, (-self.spikes) * Gmax)
            u[:] = (u * (self.spikes + 1))
            # print(self.spikes)
            return u

        return step

dt = 0.1    # ms
num_steps = 200
t = np.arange(0,dt*num_steps,dt)
data = np.zeros([2,num_steps])
current = 10

with nengo.Network() as model:
    layer = nengo.Node(SpikingNeuronsWithSynapticInput(2,5,np.array([[0,0],[0.1,0]]),np.array([[0,0],[194,0]]),1,-1,bias=np.array([current,0])))
    probe = nengo.Probe(layer)

with nengo.Simulator(model, dt=dt / 1000) as sim:
    for i in range(num_steps):
        sim.step()
        data[:,i] = sim.data[probe][i,:]

plt.figure()
plt.plot(t,data[0,:])
plt.plot(t,data[1,:])

plt.show()
