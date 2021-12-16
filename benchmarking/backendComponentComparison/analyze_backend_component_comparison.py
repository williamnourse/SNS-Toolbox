"""
Analyze the data we collected in the component comparison test
William Nourse
December 16th 2021
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

test_params = pickle.load(open('test_params.p','rb'))
num_neurons = test_params['numNeurons']

data = []
data.append(pickle.load(open('10_neurons.p', 'rb')))
data.append(pickle.load(open('21_neurons.p', 'rb')))
data.append(pickle.load(open('46_neurons.p', 'rb')))
data.append(pickle.load(open('100_neurons.p', 'rb')))
data.append(pickle.load(open('215_neurons.p', 'rb')))
data.append(pickle.load(open('464_neurons.p', 'rb')))
data.append(pickle.load(open('1000_neurons.p', 'rb')))
data.append(pickle.load(open('2154_neurons.p', 'rb')))
data.append(pickle.load(open('4641_neurons.p', 'rb')))
data.append(pickle.load(open('10000_neurons.p', 'rb')))

full_spiking_mean = np.zeros(len(num_neurons))
full_no_delay_mean = np.zeros(len(num_neurons))
full_non_spiking_mean = np.zeros(len(num_neurons))
realistic_spiking_mean = np.zeros(len(num_neurons))
realistic_no_delay_mean = np.zeros(len(num_neurons))
realistic_non_spiking_mean = np.zeros(len(num_neurons))

full_spiking_var = np.zeros(len(num_neurons))
full_no_delay_var = np.zeros(len(num_neurons))
full_non_spiking_var = np.zeros(len(num_neurons))
realistic_spiking_var = np.zeros(len(num_neurons))
realistic_no_delay_var = np.zeros(len(num_neurons))
realistic_non_spiking_var = np.zeros(len(num_neurons))

full_spiking_std = np.zeros(len(num_neurons))
full_no_delay_std = np.zeros(len(num_neurons))
full_non_spiking_std = np.zeros(len(num_neurons))
realistic_spiking_std = np.zeros(len(num_neurons))
realistic_no_delay_std = np.zeros(len(num_neurons))
realistic_non_spiking_std = np.zeros(len(num_neurons))

for i in range(len(num_neurons)):
    full_spiking_mean[i] = np.mean(data[i]['fullSpiking'])
    full_no_delay_mean[i] = np.mean(data[i]['fullNoDelay'])
    full_non_spiking_mean[i] = np.mean(data[i]['fullNonSpiking'])
    realistic_spiking_mean[i] = np.mean(data[i]['realisticSpiking'])
    realistic_no_delay_mean[i] = np.mean(data[i]['realisticNoDelay'])
    realistic_non_spiking_mean[i] = np.mean(data[i]['realisticNonSpiking'])

    full_spiking_var[i] = np.var(data[i]['fullSpiking'])
    full_no_delay_var[i] = np.var(data[i]['fullNoDelay'])
    full_non_spiking_var[i] = np.var(data[i]['fullNonSpiking'])
    realistic_spiking_var[i] = np.var(data[i]['realisticSpiking'])
    realistic_no_delay_var[i] = np.var(data[i]['realisticNoDelay'])
    realistic_non_spiking_var[i] = np.var(data[i]['realisticNonSpiking'])

    full_spiking_std[i] = np.std(data[i]['fullSpiking'])
    full_no_delay_std[i] = np.std(data[i]['fullNoDelay'])
    full_non_spiking_std[i] = np.std(data[i]['fullNonSpiking'])
    realistic_spiking_std[i] = np.std(data[i]['realisticSpiking'])
    realistic_no_delay_std[i] = np.std(data[i]['realisticNoDelay'])
    realistic_non_spiking_std[i] = np.std(data[i]['realisticNonSpiking'])

plt.figure()
plt.plot(num_neurons,full_spiking_mean,color='blue',label='Full Spiking Model')
plt.fill_between(num_neurons,full_spiking_mean-full_spiking_var,full_spiking_mean+full_spiking_var,color='blue',alpha=0.1)
plt.plot(num_neurons,full_no_delay_mean,color='orange',label='Full No Delay Model')
plt.fill_between(num_neurons,full_no_delay_mean-full_no_delay_var,full_no_delay_mean+full_no_delay_var,color='orange',alpha=0.1)
plt.plot(num_neurons,full_non_spiking_mean,color='green',label='Full Non Spiking Model')
plt.fill_between(num_neurons,full_non_spiking_mean-full_non_spiking_var,full_non_spiking_mean+full_non_spiking_var,color='green',alpha=0.1)
plt.legend()
plt.xlabel('Number of Neurons')
plt.yscale('log')
plt.xscale('log')
plt.ylabel('Average Time per Step (s)')
plt.title('Fully Connected Networks')

plt.figure()
plt.plot(num_neurons,realistic_spiking_mean,color='blue',label='Realistic Spiking Model')
plt.fill_between(num_neurons,realistic_spiking_mean-realistic_spiking_var,realistic_spiking_mean+full_spiking_var,color='blue',alpha=0.1)
plt.plot(num_neurons,realistic_no_delay_mean,color='orange',label='Realistic No Delay Model')
plt.fill_between(num_neurons,realistic_no_delay_mean-realistic_no_delay_var,realistic_no_delay_mean+realistic_no_delay_var,color='orange',alpha=0.1)
plt.plot(num_neurons,realistic_non_spiking_mean,color='green',label='Realistic Non Spiking Model')
plt.fill_between(num_neurons,realistic_non_spiking_mean-realistic_non_spiking_var,realistic_non_spiking_mean+realistic_non_spiking_var,color='green',alpha=0.1)
plt.legend()
plt.xlabel('Number of Neurons')
plt.ylabel('Average Time per Step (s)')
plt.yscale('log')
plt.xscale('log')
plt.title('Realistically Connected Networks')

plt.show()

plt.show()
