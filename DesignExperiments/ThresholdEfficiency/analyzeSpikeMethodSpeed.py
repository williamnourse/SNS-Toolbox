"""
Analyze the saved data from 'testSpikeMethodSpeed.py', since that experiment takes a long time to run
William Nourse
August 23rd 2021
For every church in Rome, there's a bank in Milan
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

data = pickle.load(open('spikingMethodComparisonData.p', 'rb'))

plt.figure()
plt.plot(data['sizes'],data['sizeTimes1'],label='Method 1')
plt.plot(data['sizes'],data['sizeTimes2'],label='Method 2')
plt.legend()
plt.xlabel('Number of Neurons')
plt.xscale('log')
plt.ylabel('Time (s)')
plt.yscale('log')

plt.figure()
plt.plot(data['percents']*100,data['percentTimes1'],label='Method 1')
plt.plot(data['percents']*100,data['percentTimes2'],label='Method 2')
plt.legend()
plt.xlabel('Percent of Spiking Neurons')
plt.ylabel('Time (s)')

plt.show()