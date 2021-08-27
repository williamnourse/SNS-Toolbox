from sns_toolbox.design.neurons import NonSpikingNeuron, SpikingNeuron, Population
from sns_toolbox.design.synapses import NonSpikingSynapse
from sns_toolbox.design.networks import Network

simple = NonSpikingNeuron()
slow = NonSpikingNeuron(membraneCapacitance=50)
transmit = NonSpikingSynapse(name='Transmit')
inhibit = NonSpikingSynapse(name='Inhibit', relativeReversalPotential=-40)
modulate = NonSpikingSynapse(name='Mod', relativeReversalPotential=0)

smallNet = Network(name='SmallNet')
smallNet.addNeuron(simple,suffix='A',color='blue')
smallNet.addNeuron(simple,suffix='B')
smallNet.addNeuron(slow,color='orange')
smallNet.addSynapse(transmit,0,2,viewLabel=True)
smallNet.addSynapse(transmit,1,2,viewLabel=True)
smallNet.renderGraph(view=True)

bigNet = Network(name='BigNet')
bigNet.addNeuron(simple,suffix='Origin')
bigNet.addNeuron(simple,suffix='Modulate',color='indianred')
bigNet.addNetwork(smallNet,color='teal')
bigNet.addSynapse(inhibit,0,1,viewLabel=True)
bigNet.addSynapse(modulate,1,2,viewLabel=True)
bigNet.renderGraph(view=True)