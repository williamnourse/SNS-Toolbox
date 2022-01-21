"""
Connect populations of neurons with patterns more complicated than simple 'all-to-all' connectivity
William Nourse
January 12, 2022
"""

from sns_toolbox.design.connections import NonSpikingPatternConnection
from sns_toolbox.design.networks import Network
from sns_toolbox.design.neurons import NonSpikingNeuron

vector_kernel = [-0.25,-0.25,1.0,-0.25,-0.25]
matrix_kernel = [[-0.125,-0.125,-0.125],
                 [-0.125,1.0,-0.125],
                 [-0.125,-0.125,-0.125]]
matrix_kernel_w_zeros = [[0.0,-0.25,0.0],
                         [-0.25,1.0,-0.25],
                         [0.0,-0.25,0.0]]

vector_connection = NonSpikingPatternConnection(vector_kernel)
matrix_connection = NonSpikingPatternConnection(matrix_kernel)
matrix_w_zeros_connection = NonSpikingPatternConnection(matrix_kernel_w_zeros)

neuron_type = NonSpikingNeuron()
net = Network(name='Tutorial 6 Network')
net.add_population(neuron_type,5,name='Source')
net.add_population(neuron_type,5,name='Dest')

net.add_connection(vector_connection,'Source','Dest')

net.render_graph(view=True)

print('Done')
