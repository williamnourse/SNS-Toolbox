import numpy as np
import torch
import time

from sns_toolbox.networks import Network
from sns_toolbox.neurons import SpikingNeuron
from sns_toolbox.connections import SpikingSynapse

from sns_toolbox import backends

backend = 2 # 0: manual
            # 1: Numpy
            # 2: Torch
            # 3: Sparse
cpu = True

theoretical_max = 200000
start_size = 10
start_time = time.time()
def design_network(size):
    print('Designing. %i sec elapsed'%(time.time()-start_time))
    net = Network()
    numIns = int(0.08 * size) + 1
    numOuts = int(0.12 * size)
    numSyn = int(np.sqrt(size))
    numRest = int(size) - numIns - numSyn - numOuts
    current = 10.0
    spike = SpikingNeuron(name='m<0', threshold_proportionality_constant=-1, color='aquamarine')
    spikeBias = SpikingNeuron(name='bias', threshold_proportionality_constant=-1, color='aquamarine', bias=current)
    spikeExcite = SpikingSynapse(name='Excitatory Spiking')
    net.add_population(spike, [numIns], name='ins')  # in puppy, num_inputs = 8% of network
    net.add_population(spikeBias, [numOuts], name='outs')  # in puppy, num_outputs = 12% of network
    net.add_population(spikeBias, [numSyn], name='connected')  # in puppy, numSyn = shape
    net.add_population(spikeBias, [numRest], name='rest')  # rest of the network
    net.add_connection(spikeExcite, 'connected', 'connected')
    net.add_input('ins')
    net.add_output('outs')
    return net

def build_network(net,backend,cpu):
    print('Building. %i sec elapsed'%(time.time()-start_time))
    if cpu:
        device = 'cpu'
    else:
        device = 'cuda'
    if backend == 0:
        model = backends.SNS_Iterative(net)
    elif backend == 1:
        model = backends.SNS_Numpy(net)
    elif backend == 2:
        model = backends.SNS_Torch(net, device=device)
    else:
        model = backends.SNS_Sparse(net, device=device)
    return model, device

def run_network(model,backend,device):
    print('Running. %i sec elapsed'%(time.time()-start_time))
    if backend > 1: # torch-based
        inp = torch.tensor([1.0],device=device)
    else:
        inp = np.array([1.0])
    out = model.forward(inp)

def process(size,backend,cpu):
    net = design_network(size)
    model, device = build_network(net,backend,cpu)
    run_network(model,backend,device)

def main_loop(start_size,max_size,backend,cpu):
    sizes = np.linspace(start_size,max_size,num=4).astype(int)
    i = 0
    failed = False
    while (not failed) and (i < len(sizes)):
        if (i > 0) and sizes[i] == sizes[i-1]:
            print('Passing Duplicate')
        else:
            print(sizes[i])
            try:
                process(sizes[i],backend,cpu)
            except:
                failed = True
        i += 1
    if failed:

        new_start = sizes[i-2]
        new_max = sizes[i-1]
        if new_start == start_size and new_max == max_size:
            print('All Done. Largest Network Size is %i neurons. %i elapsed sec'%(new_start,time.time()-start_time))
        else:
            print('Failed at size %i, trying again in new range. %i elapsed sec' % (sizes[i - 1],time.time()-start_time))
            main_loop(new_start,new_max,backend,cpu)
    else:
        print('All Done. Largest Network Size is %i neurons. %i elapsed sec'%(max_size,time.time()-start_time))

def main_loop_increment(start_size,max_size,backend,cpu):
    size = start_size
    failed = False
    while (not failed) and (size <= max_size):
        print(size)
        try:
            process(size,backend,cpu)
        except:
            failed = True
        size += 500
    if failed:
        print('All Done. Largest Network Size is %i neurons. %i elapsed sec'%(size-500,time.time()-start_time))
    else:
        print('All Done. Largest Network Size is %i neurons. %i elapsed sec'%(max_size,time.time()-start_time))

# main_loop(start_size,theoretical_max,backend,cpu)
main_loop_increment(start_size,theoretical_max+10,backend,cpu)