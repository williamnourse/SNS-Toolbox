"""
Collect data about the speed of one backend (numpy) as we change the number of neurons and the number of components
William Nourse
December 15th 2021
"""
import numpy as np
import time
import pickle

from sns_toolbox.design.networks import Network
from sns_toolbox.design.neurons import NonSpikingNeuron, SpikingNeuron
from sns_toolbox.design.connections import NonSpikingSynapse, SpikingSynapse

from sns_toolbox.simulate.backends import SNS_Numpy, SNS_Numpy_Non_Spiking, SNS_Numpy_No_Delay

def build_full_spiking_net(num_neurons,dt):
    net = Network()
    neuron_type = SpikingNeuron(bias=20.0)
    synapse_type = SpikingSynapse(transmission_delay=5)

    net.add_population(neuron_type,num_neurons=num_neurons)
    net.add_synapse(synapse_type,0,0)
    num_inputs = max(1,int(0.1*num_neurons))
    for i in range(num_inputs):
        net.add_input(0)
    net.add_output(0)
    net.add_output(0,spiking=True)

    model = SNS_Numpy(net,dt=dt)

    return model, num_inputs


def build_full_no_delay_net(num_neurons,dt):
    net = Network()
    neuron_type = SpikingNeuron(bias=20.0)
    synapse_type = SpikingSynapse()

    net.add_population(neuron_type, num_neurons=num_neurons)
    net.add_synapse(synapse_type, 0, 0)
    num_inputs = max(1, int(0.1 * num_neurons))
    for i in range(num_inputs):
        net.add_input(0)
    net.add_output(0)
    net.add_output(0,spiking=True)

    model = SNS_Numpy_No_Delay(net,dt=dt)

    return model, num_inputs


def build_full_non_spiking_net(num_neurons,dt):
    net = Network()
    neuron_type = NonSpikingNeuron(bias=20.0)
    synapse_type = NonSpikingSynapse()

    net.add_population(neuron_type, num_neurons=num_neurons)
    net.add_synapse(synapse_type, 0, 0)
    num_inputs = max(1, int(0.1 * num_neurons))
    for i in range(num_inputs):
        net.add_input(0)
    net.add_output(0)
    net.add_output(0,spiking=True)

    model = SNS_Numpy_Non_Spiking(net,dt=dt)

    return model, num_inputs

def build_realistic_spiking_net(num_neurons,dt):
    num_inputs = max(1,int(0.08*num_neurons))
    num_outputs = max(1,int(0.12*num_neurons))
    num_connected = max(1,int(np.sqrt(num_neurons)))
    num_rest = max(1,num_neurons-num_connected-num_outputs)

    net = Network()
    neuron_type = SpikingNeuron(bias=20.0)
    synapse_type = SpikingSynapse(transmission_delay=5)

    # Outputs and Inputs
    for i in range(num_outputs):
        net.add_neuron(neuron_type)
        net.add_output(i)
        net.add_output(i,spiking=True)
        if i < num_inputs:
            net.add_input(i)

    # Synaptic population
    net.add_population(neuron_type,num_connected,name='Pop')
    net.add_synapse(synapse_type,'Pop','Pop')

    # Rest of the neurons
    net.add_population(neuron_type,num_rest)

    model = SNS_Numpy(net,dt=dt)

    return model, num_inputs

def build_realistic_no_delay_net(num_neurons,dt):
    num_inputs = max(1,int(0.08*num_neurons))
    num_outputs = max(1,int(0.12*num_neurons))
    num_connected = max(1,int(np.sqrt(num_neurons)))
    num_rest = max(1,num_neurons-num_connected-num_outputs)

    net = Network()
    neuron_type = SpikingNeuron(bias=20.0)
    synapse_type = SpikingSynapse()

    # Outputs and Inputs
    for i in range(num_outputs):
        net.add_neuron(neuron_type)
        net.add_output(i)
        net.add_output(i,spiking=True)
        if i < num_inputs:
            net.add_input(i)

    # Synaptic population
    net.add_population(neuron_type,num_connected,name='Pop')
    net.add_synapse(synapse_type,'Pop','Pop')

    # Rest of the neurons
    net.add_population(neuron_type,num_rest)

    model = SNS_Numpy_No_Delay(net,dt=dt)

    return model, num_inputs


def build_realistic_non_spiking_net(num_neurons,dt):
    num_inputs = max(1, int(0.08 * num_neurons))
    num_outputs = max(1, int(0.12 * num_neurons))
    num_connected = max(1, int(np.sqrt(num_neurons)))
    num_rest = max(1, num_neurons - num_connected - num_outputs)

    net = Network()
    neuron_type = NonSpikingNeuron(bias=20.0)
    synapse_type = NonSpikingSynapse()

    # Outputs and Inputs
    for i in range(num_outputs):
        net.add_neuron(neuron_type)
        net.add_output(i)
        if i < num_inputs:
            net.add_input(i)

    # Synaptic population
    net.add_population(neuron_type, num_connected, name='Pop')
    net.add_synapse(synapse_type, 'Pop', 'Pop')

    # Rest of the neurons
    net.add_population(neuron_type, num_rest)

    model = SNS_Numpy_Non_Spiking(net,dt=dt)

    return model, num_inputs

"""
Simulate the networks
"""
dt = 0.01
num_steps = 1000
num_samples = 50
num_neurons = np.logspace(1,4,num=num_samples)
num_neurons = num_neurons.astype(int)
test_start = time.time()
test_params = {'numSamples': num_samples,
               'numSteps': num_steps,
               'numNeurons': num_neurons}
pickle.dump(test_params, open('test_params.p', 'wb'))
for num in num_neurons:
    print('{0} Neurons, Full Spiking. {1} Total Seconds'.format(num,time.time()-test_start) )
    # Full Spiking
    model, num_inputs = build_full_spiking_net(num,dt)
    inputs = np.zeros(num_inputs)
    full_spiking_data = []
    for t in range(num_steps):
        start = time.time()
        _ = model.forward(inputs)
        end = time.time()
        full_spiking_data.append(end-start)
    # filename = "full_spiking_" + str(num) + "_neurons.p"
    # pickle.dump(data,open(filename,'wb'))

    print('{0} Neurons, Full No Delay. {1} Total Seconds'.format(num, time.time() - test_start))
    # Full No Delay
    model, num_inputs = build_full_no_delay_net(num, dt)
    inputs = np.zeros(num_inputs)
    full_no_delay_data = []
    for t in range(num_steps):
        start = time.time()
        _ = model.forward(inputs)
        end = time.time()
        full_no_delay_data.append(end - start)
    # filename = "full_no_delay_" + str(num) + "_neurons.p"
    # pickle.dump(data, open(filename, 'wb'))

    print('{0} Neurons, Full Non Spiking. {1} Total Seconds'.format(num, time.time() - test_start))
    # Full Non Spiking
    model, num_inputs = build_full_non_spiking_net(num, dt)
    inputs = np.zeros(num_inputs)
    full_non_spiking_data = []
    for t in range(num_steps):
        start = time.time()
        _ = model.forward(inputs)
        end = time.time()
        full_non_spiking_data.append(end - start)
    # filename = "full_non_spiking_" + str(num) + "_neurons.p"
    # pickle.dump(data, open(filename, 'wb'))

    print('{0} Neurons, Realistic Spiking. {1} Total Seconds'.format(num, time.time() - test_start))
    # Realistic Spiking
    model, num_inputs = build_realistic_spiking_net(num, dt)
    inputs = np.zeros(num_inputs)
    realistic_spiking_data = []
    for t in range(num_steps):
        start = time.time()
        _ = model.forward(inputs)
        end = time.time()
        realistic_spiking_data.append(end - start)
    # filename = "realistic_spiking_" + str(num) + "_neurons.p"
    # pickle.dump(data, open(filename, 'wb'))

    print('{0} Neurons, Realistic No Delay. {1} Total Seconds'.format(num, time.time() - test_start))
    # Realistic No Delay
    model, num_inputs = build_realistic_no_delay_net(num, dt)
    inputs = np.zeros(num_inputs)
    realistic_no_delay_data = []
    for t in range(num_steps):
        start = time.time()
        _ = model.forward(inputs)
        end = time.time()
        realistic_no_delay_data.append(end - start)
    # filename = "realistic_no_delay_" + str(num) + "_neurons.p"
    # pickle.dump(data, open(filename, 'wb'))

    print('{0} Neurons, Realistic Non Spiking. {1} Total Seconds'.format(num, time.time() - test_start))
    # Realistic Non Spiking
    model, num_inputs = build_realistic_non_spiking_net(num, dt)
    inputs = np.zeros(num_inputs)
    realistic_non_spiking_data = []
    for t in range(num_steps):
        start = time.time()
        _ = model.forward(inputs)
        end = time.time()
        realistic_non_spiking_data.append(end - start)

    data = {'fullSpiking': full_spiking_data,
            'fullNoDelay': full_no_delay_data,
            'fullNonSpiking': full_non_spiking_data,
            'realisticSpiking': realistic_spiking_data,
            'realisticNoDelay': realistic_no_delay_data,
            'realisticNonSpiking': realistic_non_spiking_data}
    filename = str(num) + "_neurons.p"
    pickle.dump(data, open(filename, 'wb'))

print('Done in {0} seconds'.format(time.time()-test_start))

