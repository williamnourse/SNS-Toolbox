"""
Classes which can be used to construct synthetic nervous systems of basic, non-spiking neurons
William Nourse
May 6, 2021
There are no strings on me
"""

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
IMPORTS
"""

import copy
from graphviz import Digraph

# SVG standard colors for graphviz
colors = {'aliceblue','antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque',	'black', 'blanchedalmond',
          'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral',
          'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray',
          'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred',
          'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet',
          'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen',
          'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'gray', 'grey', 'green', 'greenyellow', 'honeydew',
          'hotpink', 'indianred', 'indigo',	'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen',	'lemonchiffon',
          'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey',
          'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey',
          'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon', 'mediumaquamarine',
          'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen',
          'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite',
          'navy', 'oldlace', 'olive', 'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen',
          'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple',
          'red', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna',
          'silver', 'skyblue', 'slateblue', 'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'teal',
          'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'white', 'whitesmoke', 'yellow', 'yellowgreen'}

# Standard SVG colors which need white text instead of black
colorsWhiteText = {'black', 'blue', 'blueviolet', 'brown', 'darkblue', 'darkmagenta', 'darkolivegreen', 'darkred',
          'darkslateblue', 'darkslategray', 'darkslategrey', 'darkviolet', 'dimgray', 'dimgrey', 'gray', 'grey', 'green',
          'indigo', 'maroon', 'mediumblue', 'midnightblue', 'navy', 'olive', 'purple', 'saddlebrown', 'teal'}
"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
NEURON MODELS
"""

class Neuron:
    def __init__(self, name='Neuron', membraneCapacitance=5, membraneConductance=1, bias=0, color='white'):
        self.name = name    # Name of this neuron preset
        self.membraneCapacitance = membraneCapacitance        # Membrane capacitance (nF)
        self.membraneConductance = membraneConductance        # Membrane conductance (uS)
        self.bias = bias  # Constant applied current (nA)
        self.color = color
        if color in colorsWhiteText:
            self.fontColor = 'white'
        else:
            self.fontColor = 'black'

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SYNAPSE MODELS
"""

class Synapse:
    def __init__(self, name='Synapse', maxConductance=1, relativeReversalPotential=40):
        self.name = name
        self.Gmax = maxConductance
        self.deltaE = relativeReversalPotential

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
NETWORKS
"""

class Network:
    def __init__(self,name='Network'):
        self.neurons = []
        self.synapses = []
        self.name = name
        self.graph = Digraph(filename=(self.name+'.gv'))

    def getNumNeurons(self):
        return len(self.neurons)

    def getNumSynapses(self):
        return len(self.synapses)

    def addNeuron(self, neuronType, suffix=None, color=None):
        self.neurons.append(copy.copy(neuronType))
        if suffix is not None:
            self.neurons[self.getNumNeurons()-1].name += suffix
        if color is not None:
            self.neurons[self.getNumNeurons() - 1].color = color
            if color in colorsWhiteText:
                self.neurons[self.getNumNeurons()-1].fontColor = 'white'
            else:
                self.neurons[self.getNumNeurons()-1].fontColor = 'black'
        self.graph.node(str(self.getNumNeurons()-1),self.neurons[self.getNumNeurons()-1].name, style='filled',
                        fillcolor=self.neurons[self.getNumNeurons()-1].color,fontcolor=self.neurons[self.getNumNeurons()-1].fontColor)

    def addSynapse(self, synapseType, source, destination, viewLabel=False):
        self.synapses.append(copy.copy(synapseType))
        self.synapses[self.getNumSynapses() - 1].source = source
        self.synapses[self.getNumSynapses() - 1].destination = destination
        if viewLabel:
            self.synapses[self.getNumSynapses() - 1].label = self.synapses[self.getNumSynapses() - 1].name
        else:
            self.synapses[self.getNumSynapses() - 1].label = None
        if self.synapses[self.getNumSynapses() - 1].deltaE > 0:
            style = 'invempty'
        elif self.synapses[self.getNumSynapses() - 1].deltaE < 0:
            style = 'dot'
        else:
            style = 'odot'
        self.graph.edge(str(self.synapses[self.getNumSynapses() - 1].source),
                        str(self.synapses[self.getNumSynapses() - 1].destination), arrowhead=style,
                        label=self.synapses[self.getNumSynapses() - 1].label)

    def addSubNetwork(self, network, color=None):
        numNeurons = self.getNumNeurons()
        for neuron in network.neurons:
            self.addNeuron(neuronType=neuron,color=color)
        for synapse in network.synapses:
            if synapse.label is None:
                self.addSynapse(synapseType=synapse, source=(synapse.source + numNeurons),
                                destination=(synapse.destination + numNeurons))
            else:
                self.addSynapse(synapseType=synapse, source=(synapse.source + numNeurons),
                                destination=(synapse.destination + numNeurons),viewLabel=True)

    def renderGraph(self,format='png',view=False):
        self.graph.format = format
        self.graph.render(view=view)

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
TEST
"""

simple = Neuron()
slow = Neuron(membraneCapacitance=50)
transmit = Synapse(name='Transmit')
inhibit = Synapse(name='Inhibit', relativeReversalPotential=-40)
modulate = Synapse(name='Mod', relativeReversalPotential=0)

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
bigNet.addSubNetwork(smallNet,color='teal')
bigNet.addSynapse(inhibit,0,1,viewLabel=True)
bigNet.addSynapse(modulate,1,2,viewLabel=True)
bigNet.renderGraph(view=True)
