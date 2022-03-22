import numpy as np
import torch
import time

from sns_toolbox.design.networks import Network
from sns_toolbox.design.neurons import NonSpikingNeuron
from sns_toolbox.design.connections import NonSpikingSynapse

from sns_toolbox.simulate import backends

backend = 3 # 0: manual
            # 1: Numpy
            # 2: Torch
            # 3: Sparse
cpu = True
variant = 0 # 0: full model
            # 1: No delay
            # 2: non-spiking

theoretical_max = 200000
start_size = 10
start_time = time.time()
def design_network(size):
    print('Designing. %i sec elapsed'%(time.time()-start_time))
    neuron_type = NonSpikingNeuron()
    synapse_type = NonSpikingSynapse()
    net = Network()
    net.add_population(neuron_type,[size])
    net.add_connection(synapse_type,0,0)
    net.add_input(0)
    return net

def build_network(net,backend,cpu,variant):
    print('Building. %i sec elapsed'%(time.time()-start_time))
    if cpu:
        device = 'cpu'
    else:
        device = 'cuda'
    if backend == 0:
        if variant == 0:
            model = backends.SNS_Manual(net)
        elif variant == 1:
            model = backends.SNS_Manual(net,delay=False)
        else:
            model = backends.SNS_Manual(net,spiking=False)
    elif backend == 1:
        if variant == 0:
            model = backends.SNS_Numpy(net)
        elif variant == 1:
            model = backends.SNS_Numpy(net,delay=False)
        else:
            model = backends.SNS_Numpy(net,spiking=False)
    elif backend == 2:
        if variant == 0:
            model = backends.SNS_Torch(net,device=device)
        elif variant == 1:
            model = backends.SNS_Torch(net,device=device,delay=False)
        else:
            model = backends.SNS_Torch(net,device=device,spiking=False)
    else:
        if variant == 0:
            model = backends.SNS_Sparse(net,device=device)
        elif variant == 1:
            model = backends.SNS_Sparse(net,device=device,delay=False)
        else:
            model = backends.SNS_Sparse(net,device=device,spiking=False)
    return model, device

def run_network(model,backend,device):
    print('Running. %i sec elapsed'%(time.time()-start_time))
    if backend > 1: # torch-based
        inp = torch.tensor([1.0],device=device)
    else:
        inp = np.array([1.0])
    out = model.forward(inp)

def process(size,backend,cpu,variant):
    net = design_network(size)
    model, device = build_network(net,backend,cpu,variant)
    run_network(model,backend,device)

def main_loop(start_size,max_size,backend,cpu,variant):
    sizes = np.linspace(start_size,max_size,num=4).astype(int)
    i = 0
    failed = False
    while (not failed) and (i < len(sizes)):
        if (i > 0) and sizes[i] == sizes[i-1]:
            print('Passing Duplicate')
        else:
            print(sizes[i])
            try:
                process(sizes[i],backend,cpu,variant)
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
            main_loop(new_start,new_max,backend,cpu,variant)
    else:
        print('All Done. Largest Network Size is %i neurons. %i elapsed sec'%(max_size,time.time()-start_time))

def main_loop_increment(start_size,max_size,backend,cpu,variant):
    size = start_size
    failed = False
    while (not failed) and (size < max_size):
        print(size)
        try:
            process(size,backend,cpu,variant)
        except:
            failed = True
        size += 500
    if failed:
        print('All Done. Largest Network Size is %i neurons. %i elapsed sec'%(size-500,time.time()-start_time))
    else:
        print('All Done. Largest Network Size is %i neurons. %i elapsed sec'%(max_size,time.time()-start_time))

#main_loop(start_size,theoretical_max,backend,cpu,variant)
main_loop_increment(start_size,theoretical_max+10,backend,cpu,variant)