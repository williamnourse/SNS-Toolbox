"""
Analyze the saved data from 'testSpikeResetMechanismSpeed.py', since that experiment takes a long time to run
William Nourse
August 24th 2021
That feels weird, but I'll allow it
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

data = pickle.load(open('resetMethodComparisonData.p', 'rb'))

plt.figure()
plt.plot(data['sizes'],data['sizeTimesIt'],label='Iterative')
plt.plot(data['sizes'],data['sizeTimesVec'],label='Vector')
plt.legend()
plt.xlabel('Number of Neurons')
plt.xscale('log')
plt.ylabel('Time (s)')
plt.yscale('log')

plt.figure()
plt.plot(data['percents']*100,data['percentTimesIt'],label='Iterative')
plt.plot(data['percents']*100,data['percentTimesVec'],label='Vector')
plt.legend()
plt.xlabel('Percent of Spiking Neurons')
plt.ylabel('Time (s)')

plt.show()