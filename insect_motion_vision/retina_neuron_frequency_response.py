"""
Implementation of a single retinal neuron, and measurement of the frequency response and critical flicker fusion point
William Nourse
January 6th, 2022
"""
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from sns_toolbox.design.neurons import NonSpikingNeuron
from sns_toolbox.design.networks import Network

from sns_toolbox.simulate.backends import __SNS_Numpy_Full__

Iapp = 20.0
R = 20.0
Gm = 1.0
Cm = 5.0
dt = 0.01  # ms
tMax = 1   # sec
freq = 30

t = np.arange(start=0.0,stop=tMax,step=dt/1000)
w = np.logspace(-2,3,num=50)

inp = np.zeros([len(w),len(t)])
for row in range(len(w)):
    # for col in range(len(t)):
    inp[row,:] = Iapp*np.sin(w[row]*t)

# inp = Iapp*np.sin(2*np.pi*freq*t)
# inp = np.zeros_like(t)+Iapp

neuron = NonSpikingNeuron(membrane_capacitance=Cm,membrane_conductance=Gm)
net = Network(name='Retina Neuron',R=R)
net.add_neuron(neuron)
net.add_input(0)
net.add_output(0)
model = __SNS_Numpy_Full__(net, dt=dt)

voltages = np.zeros_like(t)
ratio = np.zeros_like(w)
for freq in range(len(w)):
    print(freq)
    for i in range(len(t)):
        voltages[i] = model.forward(inp[freq,i])
    in_peak = np.max(inp[freq,:])
    out_peak = np.max(voltages)
    ratio[freq] = out_peak/in_peak
dB = 10*np.log10(ratio)

plt.figure()
plt.subplot(2,1,1)
plt.plot(t,inp[-1,:])
plt.subplot(2,1,2)
plt.plot(t,voltages)

plt.figure()
plt.plot(w,dB)
plt.xscale('log')

Gm = Gm/1e6
Cm = Cm/1e9
tau = Cm/Gm

# w = np.logspace(-2,10,num=100)
neuron = signal.TransferFunction([1], [Gm*tau, Gm])
w,mag,phase = signal.bode(neuron,w=w)

plt.figure()
plt.subplot(2,1,1)
plt.semilogx(w,mag)
plt.title('Magnitude')
plt.subplot(2,1,2)
plt.semilogx(w,phase)
plt.title('Phase')

plt.show()