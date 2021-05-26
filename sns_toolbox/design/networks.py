"""
Networks store a combination of neurons and synapse, so they can be vusualized and compiled
William Nourse
May 10, 2021
You're gonna be okay!
"""

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
IMPORTS
"""

from typing import Dict, Any, List, Type, TypeVar
from numbers import Number
import copy
from graphviz import Digraph

from sns_toolbox.design.neurons import NonSpikingNeuron
from sns_toolbox.design.synapses import NonSpikingSynapse
from sns_toolbox.design.__utilities__ import validColor, setTextColor

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
BASE CLASS
"""

class NonSpikingNetwork:
    def __init__(self, name: str = 'Network', range: float = 20.0) -> None:
        """
        Constructor for base network class
        :param name: Name for this network
        """
        self.params: Dict[str, Any] = {}
        if isinstance(name,str):
            self.params['name'] = name
        else:
            raise TypeError('Name must be a string')
        self.params['R'] = range
        self.neurons: List[NonSpikingNeuron] = []
        self.synapses: List[NonSpikingSynapse] = []
        self.graph = Digraph(filename=(self.params['name']+'.gv'))

    def getNumNeurons(self) -> int:
        """
        Get the number of neurons in the whole network
        :return: length of the neuron list
        """
        return len(self.neurons)

    def getNumSynapses(self) -> int:
        """
        Get the number of synapses in the whole network
        :return: length of the synapse list
        """
        return len(self.synapses)

    def addNeuron(self, neuron: NonSpikingNeuron, suffix: str = None, color: str = None) -> None:
        if isinstance(neuron,NonSpikingNeuron):
            self.neurons.append(copy.deepcopy(neuron))
        else:
            raise TypeError('Neuron must be of type (or inherit from) NonSpikingNeuron')
        if suffix is not None:
            if isinstance(suffix,str):
                self.neurons[self.getNumNeurons() - 1].params['name'] += suffix
            else:
                raise TypeError('Suffix must be a string')
        if validColor(color):
            self.neurons[self.getNumNeurons() - 1].params['color'] = color
            self.neurons[self.getNumNeurons() - 1].params['fontColor'] = setTextColor(
                self.neurons[self.getNumNeurons() - 1].params['color'])
        self.graph.node(str(self.getNumNeurons() - 1), self.neurons[self.getNumNeurons() - 1].params['name'],
                        style='filled',
                        fillcolor=self.neurons[self.getNumNeurons() - 1].params['color'],
                        fontcolor=self.neurons[self.getNumNeurons() - 1].params['fontColor'])

    def addSynapse(self, synapse: NonSpikingSynapse, source: int,
                   destination: int, viewLabel: bool = False, offset: int = 0) -> None:
        if isinstance(synapse,NonSpikingSynapse):
            self.synapses.append(copy.deepcopy(synapse))
        else:
            raise TypeError('Synapse must be of type (or inherit from) NonSpikingSynapse')
        if not isinstance(source,int):
            raise TypeError('Source index must be an integer')
        if not isinstance(destination,int):
            raise TypeError('Destination index must be an integer')
        if not isinstance(offset,int):
            raise TypeError('Index offset must be an integer')
        if not isinstance(viewLabel,bool):
            raise TypeError('viewLabel must be of type bool')
        if source > self.getNumNeurons()-1:
            raise ValueError('Source index is outside of network size')
        if source < 0:
            raise ValueError('Source index must be >= 0')
        if destination > self.getNumNeurons()-1:
            raise ValueError('Destination index is outside of network size')
        if destination < 0:
            raise ValueError('Destination index must be >= 0')
        if (source + offset) > self.getNumNeurons()-1:
            raise ValueError('Offset makes source index out of range')
        if (offset + destination) > self.getNumNeurons()-1:
            raise ValueError('Offset makes destination index out of range')
        self.synapses[self.getNumSynapses() - 1].params['source'] = source + offset
        self.synapses[self.getNumSynapses() - 1].params['destination'] = destination + offset
        if viewLabel:
            self.synapses[self.getNumSynapses() - 1].params['label'] = self.synapses[
                self.getNumSynapses() - 1].params['name']
        else:
            self.synapses[self.getNumSynapses() - 1].params['label'] = None
        if self.synapses[self.getNumSynapses() - 1].params['relativeReversalPotential'] > 0:
            style = 'invempty'
        elif self.synapses[self.getNumSynapses() - 1].params['relativeReversalPotential'] < 0:
            style = 'dot'
        else:
            style = 'odot'
        self.graph.edge(str(self.synapses[self.getNumSynapses() - 1].params['source']),
                        str(self.synapses[self.getNumSynapses() - 1].params['destination']), arrowhead=style,
                        label=self.synapses[self.getNumSynapses() - 1].params['label'])

    def addNetwork(self, network: 'NonSpikingNetwork', color: str = None) -> None:
        if not isinstance(network,NonSpikingNetwork):
            raise TypeError('Network needs to be of type NonSpikingNetwork')
        numNeurons = self.getNumNeurons()
        for neuron in network.neurons:
            self.addNeuron(neuron=neuron, color=color)
        for synapse in network.synapses:
            if synapse.params['label'] is None:
                self.addSynapse(synapse=synapse, source=synapse.params['source'],
                                destination=synapse.params['destination'], offset=numNeurons)
            else:
                self.addSynapse(synapse=synapse, source=synapse.params['source'],
                                destination=synapse.params['destination'], offset=numNeurons, viewLabel=True)

    def renderGraph(self, format: str = 'png', view: bool = False) -> None:
        self.graph.format = format
        self.graph.render(view=view)

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SPECIFIC MODELS
"""

# TODO: Addition/Subtraction

# TODO: Multiplication

# TODO: Division

# TODO: Integration
