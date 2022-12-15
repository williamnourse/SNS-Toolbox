from sns_toolbox.connections import NonSpikingMatrixConnection
from sns_toolbox.networks import Network
from sns_toolbox.neurons import NonSpikingNeuron
# from sns_toolbox.renderer import render

import numpy as np

net = Network()

num_0 = 5
num_1 = 3
net.add_population(NonSpikingNeuron(), np.array([num_0]))
net.add_population(NonSpikingNeuron(), np.array([num_1]))

g_matrix_01 = np.random.random([num_1,num_0])
g_matrix_10 = np.random.random([num_0,num_1])

other_matrix_01 = np.zeros_like(g_matrix_01)
other_matrix_10 = np.zeros_like(g_matrix_10)

conn_01 = NonSpikingMatrixConnection(g_matrix_01,other_matrix_01,other_matrix_01,other_matrix_01)
conn_10 = NonSpikingMatrixConnection(g_matrix_10,other_matrix_10,other_matrix_10,other_matrix_10)

net.add_connection(conn_01,0,1)
net.add_connection(conn_10,1,0)

model = net.compile(backend='numpy')

g_max = model.g_max_non
print('Original conductance matrix from 0 to 1:')
print(g_matrix_01)
print('/nOriginal conductance matrix from 1 to 0:')
print(g_matrix_10)

print('/nCompiled conductance matrix:')
print(model.g_max_non)
