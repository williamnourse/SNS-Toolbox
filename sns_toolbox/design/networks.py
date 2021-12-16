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
import numpy as np

from sns_toolbox.design.neurons import Neuron, NonSpikingNeuron
from sns_toolbox.design.connections import Synapse, NonSpikingSynapse, NonSpikingTransmissionSynapse, NonSpikingModulationSynapse
from sns_toolbox.design.__utilities__ import valid_color, set_text_color

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
BASE CLASS
"""

class Network:
    # Initialization and utilities
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
        # self.inputConns = []
        self.populations = []
        self.outputs = []
        self.outputConns = []
        self.synapses = []
        self.graph = Digraph(filename=(self.params['name']+'.gv'))

    def get_num_neurons(self) -> int:
        """
        Calculate the number of neurons in the network
        :return: num_neurons
        """
        num_neurons = 0
        for pop in self.populations:
            num_neurons += pop['number']
        return num_neurons

    def get_num_synapses(self) -> int:
        """
        Calculate the number of synapses in the network. This will need to be overhauled for populations with multiple neurons
        :return: num_synapses
        """
        num_synapses = len(self.synapses)
        return num_synapses

    def get_num_populations(self) -> int:
        num_pop = len(self.populations)
        return num_pop

    def get_num_inputs(self) -> int:
        num_in = len(self.inputs)
        return num_in

    def get_num_outputs(self) -> int:
        num_out = len(self.outputs)
        return num_out

    def get_num_outputs_actual(self) -> int:
        index = 0
        for out in range(self.get_num_outputs()):
            source_pop = self.outputs[out]['source']
            num_source_neurons = self.populations[source_pop]['number']
            for num in range(num_source_neurons):
                index += 1
        return index

    def get_population_index(self, name: str) -> int:
        if not isinstance(name,str):
            raise TypeError('Name must be a valid string')
        for index in range(len(self.populations)):
            if self.populations[index]['name'] == name:
                return index
        raise ValueError('Population not found by name \'%s\'' % str(name))

    def get_input_index(self, name: str) -> int:
        if not isinstance(name,str):
            raise TypeError('Name must be a valid string')
        for index in range(len(self.inputs)):
            if self.inputs[index]['name'] == name:
                return index
        raise ValueError('Input not found by name \'%s\'' % str(name))

    def render_graph(self, imgFormat: str = 'png', view: bool = False) -> None:
        """
        Render an image of the network in the form of a directed graph (DG)
        :param imgFormat:   File extension of the resulting image
        :param view:        Flag to view the image
        :return: None
        """
        if not isinstance(view,bool):
            raise TypeError('View must be a boolean')
        self.graph.format = imgFormat
        self.graph.render(view=view,cleanup=True)

    # Construction
    def add_population(self, neuron_type: Neuron, num_neurons: int, name: str = None, color=None) -> None:
        """
        Add a neural population to the network
        :param neuron_type:  Type of neuron to add
        :param num_neurons:  Number of that neuron to include in the population
        :param name:        Name of the population
        :param color:       Color of the population in the rendered image
        :return:            None
        """
        if not isinstance(neuron_type, Neuron):
            raise TypeError('Input type is not a neuron')
        if isinstance(num_neurons, (int,np.integer)):
            if num_neurons <= 0:
                raise ValueError('num_neurons must be > 0')
        else:
            raise TypeError('num_neurons must be an integer greater than 0')
        if name is None:
            name = neuron_type.name
        elif not isinstance(name,str):
            raise TypeError('Name must be a string')
        if color is None:
            color = neuron_type.color
        elif not valid_color(color):
            warnings.warn('Specified color is not in the standard SVG set. Defaulting to base type color.')
            color = neuron_type.color
        font_color = set_text_color(color)
        self.populations.append({'type': copy.deepcopy(neuron_type),
                                 'number': int(num_neurons),
                                 'name': name,
                                 'color': color})
        if num_neurons > 1:
            self.graph.node(str(len(self.populations)-1), name,
                            style='filled',
                            shape='doublecircle',   # Populations with multiple neurons are marked with an outline
                            fillcolor=color,
                            fontcolor=font_color)
        else:
            self.graph.node(str(len(self.populations) - 1), name,
                            style='filled',
                            fillcolor=color,
                            fontcolor=font_color)

    def add_neuron(self, neuron_type, name=None, color=None) -> None:
        """
        Add a neuron to the network. Note that this is just a special case of addPopulation, which makes a population of
        1 neuron.
        :param neuron_type:  Type of neuron to add
        :param name:        Name of the neuron
        :param color:       Color of the neuron in the visual render
        :return:    None
        """
        self.add_population(neuron_type, num_neurons=1, name=name, color=color)

    def add_input(self, dest: Any, offset: Number = 0.0, linear: Number = 1.0, quadratic: Number = 0.0,
                  cubic: Number = 0.0, name: str = 'Input', color='white') -> None:
        """
        Add an input source to the network
        :param dest:        Destination this input connects to
        :param offset:      Constant offset of input values in polynomial map
        :param linear:      Linear integration_gain of input values in polynomial map
        :param quadratic:   Quadratic (^2) integration_gain of input values in polynomial map
        :param cubic:       Cubic (^3) integration_gain of input values in polynomial map
        :param name:        Name of the input node
        :param color:       Color of the input node in the visual render
        :return:    None
        """

        if not isinstance(name,str):
            raise TypeError('Name must be a string')
        if not valid_color(color):
            warnings.warn('Specified color is not in the standard SVG set. Defaulting to white.')
            color = 'white'
        if not isinstance(dest, int):
            if isinstance(dest, str):
                dest = self.get_population_index(dest)
            else:
                raise TypeError('Destination index must be an integer or name')
        if dest > (len(self.populations)-1):
            raise ValueError('Destination index is out of range')
        font_color = set_text_color(color)
        self.inputs.append({'name': name,
                            'destination': dest,
                            'offset': offset,
                            'linear': linear,
                            'quadratic': quadratic,
                            'cubic': cubic,
                            'color': color})
        self.graph.node('In'+str(len(self.inputs) - 1), name,
                        style='filled',
                        shape='invhouse',
                        fillcolor=color,
                        fontcolor=font_color)
        self.graph.edge('In'+str(len(self.inputs) - 1), str(dest))

    def add_output(self, source: Any, offset: Number = 0.0, linear: Number = 1.0, quadratic: Number = 0.0,
                   cubic: Number = 0.0, name: str = 'Output', spiking: bool = False, color: str = 'white') -> None:
        """
        Add an output node to the network
        :param source:      Source this output is connected to
        :param offset:      Constant offset of output values in polynomial map
        :param linear:      Linear integration_gain of output values in polynomial map
        :param quadratic:   Quadratic (^2) integration_gain of output values in polynomial map
        :param cubic:       Cubic (^3) integration_gain of output values in polynomial map
        :param name:        Name of the node
        :param spiking:     Flag for if this node stores voltage or spikes
        :param color:       Color of the output in the visual render
        :return: None
        """
        if isinstance(source,int):
            if source < 0:
                raise ValueError('Source must be an integer greater than or equal to 0')
        elif isinstance(source,str):
            source = self.get_population_index(source)
        else:
            raise TypeError('Source must be an integer greater than 0 or a name')
        if source > (len(self.populations)-1):
            raise ValueError('Source index is out of range')
        if not isinstance(name,str):
            raise TypeError('Name must be a string')
        if not isinstance(spiking,bool):
            raise TypeError('Spiking flag must be a boolean')
        if not valid_color(color):
            warnings.warn('Specified color is not in the standard SVG set. Defaulting to white.')
            color = 'white'
        font_color = set_text_color(color)
        self.outputs.append({'name': name,
                             'source': source,
                             'offset': offset,
                             'linear': linear,
                             'quadratic': quadratic,
                             'cubic': cubic,
                             'spiking': spiking,
                             'color': color})
        if spiking:
            self.graph.node('Out'+str(len(self.outputs) - 1), name,
                            style='filled',
                            shape='triangle',
                            fillcolor=color,
                            fontcolor=font_color)
        else:
            self.graph.node('Out' + str(len(self.outputs) - 1), name,
                            style='filled',
                            shape='house',
                            fillcolor=color,
                            fontcolor=font_color)

        self.graph.edge(str(source),'Out'+str(len(self.outputs)-1))

    def add_synapse(self, synapse_type: Synapse, source: Any,
                    destination: Any, name: str = None, view_label: bool = False) -> None:
        """
        Add a synaptic connection between two populations in the network
        :param synapse_type: Type of synapse to add
        :param source:      Index of source population in the network
        :param destination: Index of destination population in the network
        :param name:        Name of synapse
        :param view_label:   Flag to render the name on the output graph
        :return: None
        """
        if not isinstance(synapse_type, Synapse):
            raise TypeError('Synapse type must be of type Synapse (or inherit from it)')
        if not isinstance(source,int):
            if isinstance(source,str):
                source = self.get_population_index(source)
            else:
                raise TypeError('Source index must be an integer or name')
        if not isinstance(destination,int):
            if isinstance(destination, str):
                destination = self.get_population_index(destination)
            else:
                raise TypeError('Destination index must be an integer or name')
        if not isinstance(view_label, bool):
            raise TypeError('view_label must be of type bool')
        if source > (len(self.populations)-1):
            raise ValueError('Source index is outside of network size')
        if source < 0:
            raise ValueError('Source index must be >= 0')
        if not isinstance(destination,int):
            if isinstance(destination,str):
                destination = self.get_population_index(destination)
            else:
                raise TypeError('Destination index must be an integer or name')
        if destination > (len(self.populations)-1):
            raise ValueError('Destination index is outside of network size')
        if destination < 0:
            raise ValueError('Destination index must be >= 0')
        if name is None:
            label = synapse_type.name
        else:
            if isinstance(name,str):
                label = name
            else:
                raise TypeError('Name must be a string')
        self.synapses.append({'name': label,
                              'source': source,
                              'destination': destination,
                              'type': copy.deepcopy(synapse_type),
                              'view': view_label})
        if synapse_type.params['relative_reversal_potential'] > 0:
            style = 'invempty'
        elif synapse_type.params['relative_reversal_potential'] < 0:
            style = 'dot'
        else:
            style = 'odot'
        if view_label:
            self.graph.edge(str(source),
                            str(destination), arrowhead=style,
                            label=label)
        else:
            self.graph.edge(str(source),
                            str(destination), arrowhead=style)

    def add_network(self, network: 'Network', color: str = None) -> None:
        """
        Add an existing topology of inputs, outputs, and populations to the network
        :param network: Network to copy over
        :param color:   Color to render nodes in the network
        :return: None
        """
        if not isinstance(network, Network):
            raise TypeError('Network must be of type Network')


        num_inputs = len(self.inputs)
        num_populations = len(self.populations)
        num_outputs = len(self.outputs)

        if color is None:
            for population in network.populations:
                self.add_population(neuron_type=population['type'], num_neurons=population['number'],
                                    name=population['name'], color=population['color'])
            for inp in network.inputs:
                self.add_input(dest=inp['destination'] + num_populations, name=inp['name'], color=inp['color'],
                               offset=inp['offset'], linear=inp['linear'], quadratic=inp['quadratic'],
                               cubic=inp['cubic'])
            for out in network.outputs:
                self.add_output(source=out['source'] + num_populations, offset=out['offset'], linear=out['linear'],
                                quadratic=out['quadratic'], cubic=out['cubic'], name=out['name'], color=out['color'],
                                spiking=out['spiking'])
            for synapse in network.synapses:
                self.add_synapse(synapse_type=synapse['type'], source=synapse['source'] + num_populations,
                                 destination=synapse['destination']+num_populations, view_label=synapse['view'])
        else:
            if not valid_color(color):
                warnings.warn('Specified color is not in the standard SVG set. Defaulting to white.')
                color = 'white'
            for population in network.populations:
                self.add_population(neuron_type=population['type'], num_neurons=population['number'],
                                    name=population['name'], color=color)
            for inp in network.inputs:
                self.add_input(dest=inp['destination'] + num_populations, name=inp['name'], color=color,
                               offset=inp['offset'], linear=inp['linear'], quadratic=inp['quadratic'],
                               cubic=inp['cubic'])
            for out in network.outputs:
                self.add_output(source=out['source'] + num_populations, offset=out['offset'], linear=out['linear'],
                                quadratic=out['quadratic'], cubic=out['cubic'], name=out['name'], color=color,
                                spiking=out['spiking'])
            for synapse in network.synapses:
                self.add_synapse(synapse_type=synapse['type'], source=synapse['source'] + num_populations,
                                 destination=synapse['destination']+num_populations, view_label=synapse['view'])

    def copy(self):
        new_net = Network(name=self.params['name'],R=self.params['R'])
        new_net.add_network(self)

        return new_net

    # Modification

    # Deletion

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SPECIFIC MODELS
"""

class AdditionNetwork(Network):
    def __init__(self,gains,add_del_e=100,sub_del_e=-40,neuron_type=NonSpikingNeuron(),name='Add',**kwargs):
        super().__init__(name=name,**kwargs)
        num_inputs = len(gains)
        self.add_neuron(neuron_type=neuron_type, name=name + 'Sum')
        for i in range(num_inputs):
            self.add_neuron(neuron_type, name=name + 'Src' + str(i))
            gain = gains[i]
            if gain > 0:
                conn = NonSpikingTransmissionSynapse(gain=gain, relative_reversal_potential=add_del_e, R=self.params['R'])
            else:
                conn = NonSpikingTransmissionSynapse(gain=gain, relative_reversal_potential=sub_del_e, R=self.params['R'])
            self.add_synapse(conn, i + 1, name + 'Sum')

class MultiplicationNetwork(Network):
    def __init__(self,neuron_type=NonSpikingNeuron(),name='Multiply',**kwargs):
        super().__init__(name=name,**kwargs)
        self.add_neuron(neuron_type, name=name + '0')
        self.add_neuron(neuron_type, name=name + '1')
        self.add_neuron(neuron_type, name=name + 'Inter')
        self.add_neuron(neuron_type, name=name + 'Result')

        transmit = NonSpikingTransmissionSynapse(gain=1.0,R=self.params['R'])
        conductance = -self.params['R']/-1.0
        modulate_special = NonSpikingSynapse(max_conductance=conductance, relative_reversal_potential=-1.0)

        self.add_synapse(transmit, name + '0', name + 'Result')
        self.add_synapse(modulate_special, name + '1', name + 'Inter')
        self.add_synapse(modulate_special, name + 'Inter', name + 'Result')

class DivisionNetwork(Network):
    def __init__(self,gain,ratio,name='Divide',neuron_type=NonSpikingNeuron(),**kwargs):
        super().__init__(name=name,**kwargs)
        self.add_neuron(neuron_type, name=name + 'Transmit')
        self.add_neuron(neuron_type, name=name + 'Modulate')
        self.add_neuron(neuron_type, name=name + 'Results')

        transmission = NonSpikingTransmissionSynapse(gain,R=self.params['R'])
        self.add_synapse(transmission, 0, 2)

        modulation = NonSpikingModulationSynapse(ratio)
        self.add_synapse(modulation, 1, 2)

class DifferentiatorNetwork(Network):
    def __init__(self,slew_rate=1.0,name='Differentiate',tau_fast=1.0,**kwargs):
        super().__init__(name=name,**kwargs)
        fast_neuron_type = NonSpikingNeuron(membrane_capacitance=tau_fast,membrane_conductance=1.0)
        tau_slow = tau_fast + self.params['R']/slew_rate
        slow_neuron_type = NonSpikingNeuron(membrane_capacitance=tau_slow,membrane_conductance=1.0)
        add_synapse = NonSpikingTransmissionSynapse(gain=1.0,R=self.params['R'],relative_reversal_potential=40.0)
        sub_synapse = NonSpikingTransmissionSynapse(gain=-1.0,R=self.params['R'],relative_reversal_potential=-40.0)

        self.add_neuron(neuron_type=fast_neuron_type,name='Uin')
        self.add_neuron(neuron_type=fast_neuron_type,name='Ufast')
        self.add_neuron(neuron_type=slow_neuron_type,name='Uslow')
        self.add_neuron(neuron_type=fast_neuron_type,name='Uout')

        self.add_synapse(add_synapse,'Uin','Ufast')
        self.add_synapse(add_synapse, 'Uin', 'Uslow')
        self.add_synapse(add_synapse, 'Ufast', 'Uout')
        self.add_synapse(sub_synapse, 'Uslow', 'Uout')

class IntegratorNetwork(Network):
    def __init__(self, integration_gain=0.1, relative_reversal_potential=-40.0, name='Integrator', **kwargs):
        super().__init__(name=name,**kwargs)
        membrane_capacitance = 1/(2 * integration_gain)
        if relative_reversal_potential < 0.0:
            synaptic_conductance = -self.params['R']/relative_reversal_potential
        else:
            raise ValueError('Relative reversal potential (Delta E) must be less than zero for an integrator')

        neuron_type = NonSpikingNeuron(membrane_capacitance=membrane_capacitance,membrane_conductance=1.0,
                                       bias=self.params['R'])
        synapse_type = NonSpikingSynapse(max_conductance=synaptic_conductance,
                                         relative_reversal_potential=relative_reversal_potential)

        self.add_neuron(neuron_type,name='Uint')
        self.add_neuron(neuron_type)

        self.add_synapse(synapse_type,0,1)
        self.add_synapse(synapse_type,1,0)


# TODO: Adaptation
class AdaptationNetwork(Network):
    def __init__(self,ratio=0.5,name='Adaptation',neuron_type=NonSpikingNeuron(),**kwargs):
        super().__init__(name=name,**kwargs)
        relative_reversal_potential = 2*(self.params['R']**2 + 1)/self.params['R']
        conductance_fast_slow = self.params['R']/(relative_reversal_potential - self.params['R'])
        conductance_slow_fast = (self.params['R']*(ratio-1)*(relative_reversal_potential - self.params['R'] +
                                                             ratio*self.params['R']))/(ratio *
                                                                                       relative_reversal_potential *
                                                                                       (-relative_reversal_potential -
                                                                                        ratio*self.params['R']))
        fast_slow = NonSpikingSynapse(max_conductance=conductance_fast_slow,
                                      relative_reversal_potential=relative_reversal_potential)
        slow_fast = NonSpikingSynapse(max_conductance=conductance_slow_fast,
                                      relative_reversal_potential=-relative_reversal_potential)

        self.add_neuron(neuron_type,name='Uadapt')
        self.add_neuron(neuron_type)

        self.add_synapse(fast_slow,0,1)
        self.add_synapse(slow_fast,1,0)