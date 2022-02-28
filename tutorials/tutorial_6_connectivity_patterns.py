"""
Connect populations of neurons with patterns more complicated than simple 'all-to-all' connectivity
William Nourse
January 12, 2022
"""

from sns_toolbox.design.connections import NonSpikingPatternConnection
from sns_toolbox.design.networks import Network
from sns_toolbox.design.neurons import NonSpikingNeuron

from sns_toolbox.simulate.backends import SNS_Numpy

import numpy as np

max_conductance_kernel = np.array([[0.1, 0.5, 1.0, 0.5, 0.1]])
relative_reversal_potential_kernel = np.array([[-0.25,-0.25,1.0,-0.25,-0.25]])

vector_connection = NonSpikingPatternConnection(max_conductance_kernel,relative_reversal_potential_kernel)

neuron_type = NonSpikingNeuron()
net = Network(name='Tutorial 6 Network')
net.add_population(neuron_type,[5,5],name='Source')
net.add_population(neuron_type,[5,5],name='Dest')
net.add_input('Source',size=25)
net.add_output('Dest',name='Volt')
net.add_output('Dest',name='Spike', spiking=True)

net.add_connection(vector_connection,'Source','Dest')

# net.render_graph(view=True)

dt = 0.01
model = SNS_Numpy(net,dt=dt)

print('Done')
