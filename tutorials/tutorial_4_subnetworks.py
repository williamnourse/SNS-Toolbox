"""
Explore the range of pre-built functional subnetworks
William Nourse
December 3rd 2021
"""

from sns_toolbox.design.networks import Network, AdditionNetwork
from sns_toolbox.design.neurons import NonSpikingNeuron

net = Network(name='Tutorial 4 Network')

sum = AdditionNetwork([1,-1,-0.5,2])
net.addNetwork(sum)

net.renderGraph(view=True)
