"""
Explore the range of pre-built functional subnetworks
William Nourse
December 3rd 2021
"""

from sns_toolbox.networks import Network #, AdditionNetwork (This would import the code that we remake here
from sns_toolbox.neurons import NonSpikingNeuron
from sns_toolbox.connections import NonSpikingTransmissionSynapse
from sns_toolbox.networks import DivisionNetwork, MultiplicationNetwork, DifferentiatorNetwork
from sns_toolbox.networks import IntegratorNetwork

# Let's define a custom functional subnetwork 'preset', in this case a network which takes a weighted sum of inputs

class AdditionNetwork(Network): # inherit from the general 'Network' class
    def __init__(self,gains,add_del_e=100,sub_del_e=-40,neuron_type=NonSpikingNeuron(),name='Add',**kwargs):
        super().__init__(**kwargs)  # This part is important, it initializes the base class first with its keywords
        num_inputs = len(gains)
        self.add_neuron(neuron_type=neuron_type, name=name + 'Sum')  # Add a neuron which represents the sum
        for i in range(num_inputs):
            self.add_neuron(neuron_type, name=name + 'Src' + str(i))  # Add each of the input neurons
            gain = gains[i]
            if gain > 0:    # create connections differently depending on whether the integration_gain is positive or negative
                conn = NonSpikingTransmissionSynapse(gain=gain, relative_reversal_potential=add_del_e, R=self.params['R'])
            else:
                conn = NonSpikingTransmissionSynapse(gain=gain, relative_reversal_potential=sub_del_e, R=self.params['R'])
            self.add_connection(conn, i + 1, name + 'Sum')    # add the synapse to the network


# Now let's import our network into another one, as we would normally use this functionality
net = Network(name='Tutorial 4 Network')

sum_net = AdditionNetwork([1,-1,-0.5,2])
net.add_network(sum_net, color='blue')

# We can add more subnetworks
div_net = DivisionNetwork(1,0.5)
net.add_network(div_net, color='orange')

mult_net = MultiplicationNetwork()
net.add_network(mult_net, color='green')

diff_net = DifferentiatorNetwork()
net.add_network(diff_net,color='red')

int_net = IntegratorNetwork()
net.add_network(int_net,color='purple')

net.render_graph(view=True)
