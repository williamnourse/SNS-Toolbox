"""
Networks store a combination of neurons and synapse, so they can be visualized and compiled
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
import warnings

from sns_toolbox.design.neurons import Neuron
from sns_toolbox.design.connections import NonSpikingSynapse, Synapse, SpikingSynapse
from sns_toolbox.design.__utilities__ import validColor, setTextColor

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
BASE CLASS
"""

class Network:
    def __init__(self, name: str = 'Network', R: float = 20.0) -> None:
        """
        Constructor for base network class
        :param name:    Name for this network
        :param R:       Range of activity for this network (mV)
        :return:    None
        """
        self.params: Dict[str, Any] = {}
        if isinstance(name,str):
            self.params['name'] = name
        else:
            raise TypeError('Name must be a string')
        if isinstance(R,Number):
            if R > 0:
                self.params['R'] = R
            else:
                raise ValueError('R must be > 0')
        else:
            raise TypeError('R must be a number')
        self.inputs = []
        self.inputConns = []
        self.populations = []
        self.outputs = []
        self.outputConns = []
        self.synapses = []
        self.graph = Digraph(filename=(self.params['name']+'.gv'))

    def getNumNeurons(self):
        """
        Calculate the number of neurons in the network
        :return: numNeurons
        """
        numNeurons = 0
        for pop in self.populations:
            numNeurons += pop['number']
        return numNeurons

    def getNumSynapses(self):
        """
        Calculate the number of synapses in the network. This will need to be overhauled for populations with multiple neurons
        :return: numSynapses
        """
        numSynapses = len(self.synapses)
        return numSynapses

    def getNumPopulations(self):
        numPop = len(self.populations)
        return numPop

    def getNumInputs(self):
        numIn = len(self.inputs)
        return numIn

    def getNumOutputs(self):
        numOut = len(self.outputs)
        return numOut

    def addPopulation(self,neuronType: Neuron,numNeurons: int,name: str = None,color=None):
        """
        Add a neural population to the network
        :param neuronType:  Type of neuron to add
        :param numNeurons:  Number of that neuron to include in the population
        :param name:        Name of the population
        :param color:       Color of the population in the rendered image
        :return:            None
        """
        if not isinstance(neuronType,Neuron):
            raise TypeError('Input type is not a neuron')
        if isinstance(numNeurons,int):
            if numNeurons <= 0:
                raise ValueError('numNeurons must be > 0')
        else:
            raise TypeError('numNeurons must be an integer greater than 0')
        if name is None:
            name = neuronType.name
        elif not isinstance(name,str):
            raise TypeError('Name must be a string')
        if color is None:
            color = neuronType.color
        elif not validColor(color):
            warnings.warn('Specified color is not in the standard SVG set. Defaulting to base type color.')
            color = neuronType.color
        fontColor = setTextColor(color)
        self.populations.append({'type': neuronType,
                                 'number': int(numNeurons),
                                 'name': name,
                                 'color': color})
        if numNeurons > 1:
            self.graph.node(str(len(self.populations)-1), name,
                            style='filled',
                            shape='doublecircle',   # Populations with multiple neurons are marked with an outline
                            fillcolor=color,
                            fontcolor=fontColor)
        else:
            self.graph.node(str(len(self.populations) - 1), name,
                            style='filled',
                            fillcolor=color,
                            fontcolor=fontColor)

    def addNeuron(self,neuronType,name=None,color=None):
        """
        Add a neuron to the network. Note that this is just a special case of addPopulation, which makes a population of
        1 neuron.
        :param neuronType:  Type of neuron to add
        :param name:        Name of the neuron
        :param color:       Color of the neuron in the visual render
        :return:    None
        """
        self.addPopulation(neuronType,numNeurons=1,name=name,color=color)

    def addInput(self,name: str = 'Input',color='white'):
        """
        Add an input source to the network
        :param name:        Name of the input node
        :param color:       Color of the input node in the visual render
        :return:    None
        """

        if not isinstance(name,str):
            raise TypeError('Name must be a string')
        if not validColor(color):
            warnings.warn('Specified color is not in the standard SVG set. Defaulting to white.')
            color = 'white'
        fontColor = setTextColor(color)
        self.inputs.append({'name': name,
                            'color': color})
        self.graph.node('In'+str(len(self.inputs) - 1), name,
                        style='filled',
                        shape='invhouse',
                        fillcolor=color,
                        fontcolor=fontColor)

    def addOutput(self,name: str = 'Output',spiking: bool = False,color: str = 'white'):
        """
        Add an output node to the network
        :param name:        Name of the node
        :param spiking:     Flag for if this node stores voltage or spikes
        :param color:       Color of the output in the visual render
        :return: None
        """
        if not isinstance(name,str):
            raise TypeError('Name must be a string')
        if not isinstance(spiking,bool):
            raise TypeError('Spiking flag must be a boolean')
        if not validColor(color):
            warnings.warn('Specified color is not in the standard SVG set. Defaulting to white.')
            color = 'white'
        fontColor = setTextColor(color)
        self.outputs.append({'name': name,
                             'spiking': spiking,
                             'color': color})
        if spiking:
            self.graph.node('Out'+str(len(self.outputs) - 1), name,
                            style='filled',
                            shape='triangle',
                            fillcolor=color,
                            fontcolor=fontColor)
        else:
            self.graph.node('Out' + str(len(self.outputs) - 1), name,
                            style='filled',
                            shape='house',
                            fillcolor=color,
                            fontcolor=fontColor)

    def addInputConnection(self,weight: Number,source: int, dest: int, viewWeight: bool = False):
        """
        Add a weighted connection from an input node to a population in the network
        :param weight:      Weight of the connection
        :param source:      Index of source input node
        :param dest:        Index of destination population
        :param viewWeight:  Flag for the weight to be visually displayed
        :return:
        """
        if not isinstance(weight,Number):
            raise TypeError('Weight must be a number')
        if not isinstance(source,int):
            raise TypeError('Source index must be an integer')
        if source > (len(self.inputs)-1):
            raise ValueError('Source index is out of range')
        if not isinstance(dest,int):
            raise TypeError('Destination index must be an integer')
        if dest > (len(self.populations)-1):
            raise ValueError('Destination index is out of range')
        if not isinstance(viewWeight,bool):
            raise TypeError('viewWeight must be a boolean')
        self.inputConns.append({'weight': weight,
                                'source': source,
                                'destination': dest,
                                'view': viewWeight})
        if viewWeight:
            label = str(weight)
        else:
            label = None
        self.graph.edge('In'+str(source),str(dest),label=label)

    def addOutputConnection(self,weight: float,source: int, dest: int, viewWeight: bool = False):
        """
        Add a weighted connection from a population in the network to an output node
        :param weight:      Weight of the connection
        :param source:      Index of the source neuron in the network
        :param dest:        Index of the destination output node
        :param viewWeight:  Boolean flag for viewing the weight in the rendered graph
        :return: None
        """
        if not isinstance(weight,Number):
            raise TypeError('Weight must be a number')
        if not isinstance(source,int):
            raise TypeError('Source index must be an integer')
        if source > (len(self.populations)-1):
            raise ValueError('Source index is out of range')
        if not isinstance(dest,int):
            raise TypeError('Destination index must be an integer')
        if dest > (len(self.outputs)-1):
            raise ValueError('Destination index is out of range')
        if not isinstance(viewWeight,bool):
            raise TypeError('viewWeight must be a boolean')
        self.outputConns.append({'weight': weight,
                                 'source': source,
                                 'destination': dest,
                                 'view': viewWeight})
        if viewWeight:
            label = str(weight)
        else:
            label = None
        self.graph.edge(str(source),'Out'+str(dest),label=label)

    def addSynapse(self, synapseType: Synapse, source: int,
                   destination: int, name: str = None, viewLabel: bool = False) -> None:
        """
        Add a synaptic connection between two populations in the network
        :param synapseType: Type of synapse to add
        :param source:      Index of source population in the network
        :param destination: Index of destination population in the network
        :param name:        Name of synapse
        :param viewLabel:   Flag to render the name on the output graph
        :return: None
        """
        if not isinstance(synapseType,Synapse):
            raise TypeError('Synapse type must be of type Synapse (or inherit from it)')
        if not isinstance(source,int):
            raise TypeError('Source index must be an integer')
        if not isinstance(destination,int):
            raise TypeError('Destination index must be an integer')
        if not isinstance(viewLabel,bool):
            raise TypeError('viewLabel must be of type bool')
        if source > (len(self.populations)-1):
            raise ValueError('Source index is outside of network size')
        if source < 0:
            raise ValueError('Source index must be >= 0')
        if destination > (len(self.populations)-1):
            raise ValueError('Destination index is outside of network size')
        if destination < 0:
            raise ValueError('Destination index must be >= 0')
        if name is None:
            label = synapseType.name
        else:
            if isinstance(name,str):
                label = name
            else:
                raise TypeError('Name must be a string')
        self.synapses.append({'name': label,
                              'source': source,
                              'destination': destination,
                              'type': synapseType,
                              'view': viewLabel})
        if synapseType.params['relativeReversalPotential'] > 0:
            style = 'invempty'
        elif synapseType.params['relativeReversalPotential'] < 0:
            style = 'dot'
        else:
            style = 'odot'
        if viewLabel:
            self.graph.edge(str(source),
                            str(destination), arrowhead=style,
                            label=label)
        else:
            self.graph.edge(str(source),
                            str(destination), arrowhead=style)

    def addNetwork(self, network: 'Network', color: str = None) -> None:
        """
        Add an existing topology of inputs, outputs, and populations to the network
        :param network: Network to copy over
        :param color:   Color to render nodes in the network
        :return: None
        """
        if not isinstance(network, Network):
            raise TypeError('Network must be of type Network')


        numInputs = len(self.inputs)
        numPopulations = len(self.populations)
        numOutputs = len(self.outputs)

        if color is None:
            for population in network.populations:
                self.addPopulation(neuronType=population['type'],numNeurons=population['number'],name=population['name'],color=population['color'])
            for inp in network.inputs:
                self.addInput(name=inp['name'],color=inp['color'])
            for out in network.outputs:
                self.addOutput(name=out['name'],color=out['color'],spiking=out['spiking'])
            for inConn in network.inputConns:
                self.addInputConnection(inConn['weight'],inConn['source']+numInputs,inConn['destination']+numPopulations,viewWeight=inConn['view'])
            for outConn in network.outputConns:
                self.addOutputConnection(outConn['weight'],outConn['source']+numPopulations,outConn['destination']+numOutputs,viewWeight=outConn['view'])
            for synapse in network.synapses:
                self.addSynapse(synapseType=synapse['type'],source=synapse['source']+numPopulations,destination=synapse['destination']+numPopulations,viewLabel=synapse['view'])
        else:
            if not validColor(color):
                warnings.warn('Specified color is not in the standard SVG set. Defaulting to white.')
                color = 'white'
            for population in network.populations:
                self.addPopulation(neuronType=population['type'],numNeurons=population['number'],name=population['name'],color=color)
            for inp in network.inputs:
                self.addInput(name=inp['name'],color=color)
            for out in network.outputs:
                self.addOutput(name=out['name'],color=color,spiking=out['spiking'])
            for inConn in network.inputConns:
                self.addInputConnection(inConn['weight'],inConn['source']+numInputs,inConn['destination']+numPopulations,viewWeight=inConn['view'])
            for outConn in network.outputConns:
                self.addOutputConnection(outConn['weight'],outConn['source']+numPopulations,outConn['destination']+numOutputs,viewWeight=outConn['view'])
            for synapse in network.synapses:
                self.addSynapse(synapseType=synapse['type'],source=synapse['source']+numPopulations,destination=synapse['destination']+numPopulations,viewLabel=synapse['view'])

    def renderGraph(self, imgFormat: str = 'png', view: bool = False) -> None:
        """
        Render an image of the network in the form of a directed graph (DG)
        :param imgFormat:   File extension of the resulting image
        :param view:        Flag to view the image
        :return: None
        """
        if not isinstance(view,bool):
            raise TypeError('View must be a boolean')
        self.graph.format = imgFormat
        self.graph.render(view=view)

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SPECIFIC MODELS
"""

# TODO: Addition/Subtraction

# TODO: Multiplication

# TODO: Division

# TODO: Integration
