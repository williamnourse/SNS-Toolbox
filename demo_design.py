from sns_toolbox.design.neurons import NonSpikingNeuron, SpikingNeuron
from sns_toolbox.design.connections import NonSpikingSynapse
from sns_toolbox.design.networks import Network

nonSpikeType = NonSpikingNeuron(name='NonSpiking',color='blue')
spikeType = SpikingNeuron(name='Spiking',color='green')

simpleNet = Network(name='SimpleNet')
simpleNet.addNeuron(nonSpikeType,name='Single Non',color='foo')
simpleNet.addNeuron(spikeType,name='Single Spike',color='orange')
simpleNet.addInput(color='indianred')
simpleNet.addOutput(name='2 Outputs',color='mediumaquamarine')

simpleNet.addInputConnection(5.0,0,0)
simpleNet.addInputConnection(1.0,0,1,viewWeight=True)
simpleNet.addOutputConnection(5.0,0,0,viewWeight=True)
simpleNet.addOutputConnection(1.0,1,0)

transmit = NonSpikingSynapse(name='Transmit')
inhibit = NonSpikingSynapse(name='Inhibit', relativeReversalPotential=-40)
modulate = NonSpikingSynapse(name='Mod', relativeReversalPotential=0)

simpleNet.addSynapse(modulate, 0, 1, viewLabel=True)
simpleNet.addSynapse(inhibit, 1, 0)

simpleNet.renderGraph(view=True)

biggerNet = Network(name='BiggerNet')
biggerNet.addInput()
biggerNet.addOutput()
biggerNet.addNeuron(nonSpikeType,name='1')
biggerNet.addPopulation(nonSpikeType,2,name='2')
biggerNet.addSynapse(transmit,0,1)
biggerNet.addNetwork(simpleNet,color='burlywood')
biggerNet.addSynapse(transmit,0,2,viewLabel=True)
biggerNet.addSynapse(inhibit,3,1)
biggerNet.addInputConnection(1,0,0)
biggerNet.addOutputConnection(2,1,0)

biggerNet.renderGraph(view=True)