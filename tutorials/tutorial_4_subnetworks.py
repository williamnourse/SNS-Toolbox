"""
Explore the range of pre-built functional subnetworks
William Nourse
December 3rd 2021
"""

from sns_toolbox.design.networks import Network #, AdditionNetwork (This would import the code that we remake here
from sns_toolbox.design.neurons import NonSpikingNeuron
from sns_toolbox.design.connections import NonSpikingTransmissionSynapse

# Let's define a custom functional subnetwork 'preset', in this case a network which takes a weighted sum of inputs

class AdditionNetwork(Network): # inherit from the general 'Network' class
    def __init__(self,gains,add_del_e=100,sub_del_e=-40,neuron_type=NonSpikingNeuron(),name='Add',**kwargs):
        super().__init__(**kwargs)
        num_inputs = len(gains)
        self.addNeuron(neuronType=neuron_type,name=name+'Sum')
        for i in range(num_inputs):
            self.addNeuron(neuron_type,name=name+'Src'+str(i))
            gain = gains[i]
            if gain > 0:
                conn = NonSpikingTransmissionSynapse(gain=gain,relativeReversalPotential=add_del_e)
            else:
                conn = NonSpikingTransmissionSynapse(gain=gain, relativeReversalPotential=sub_del_e)
            self.addSynapse(conn,i+1,name+'Sum')


net = Network(name='Tutorial 4 Network')

sum_net = AdditionNetwork([1,-1,-0.5,2])
net.addNetwork(sum_net)

net.renderGraph(view=True)
