"""
Networks store neuron and synapse types, and the pattern of connections between them. While not capable of being
simulated directly, they are used as a template for compilation.
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

from sns_toolbox.design.neurons import Neuron, NonSpikingNeuron, SpikingNeuron
from sns_toolbox.design.connections import Connection, NonSpikingSynapse, NonSpikingTransmissionSynapse, NonSpikingModulationSynapse
from sns_toolbox.design.design_utilities import valid_color, set_text_color

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
BASE CLASS
"""

class Network:
    """
    Base class for all Networks.

    :param name: Name for this network, defaults to 'Network".
    :type name: str, optional
    :param R: Range of activity for the network, defaults to 20.0. Units are millivolts (mV).
    :type R: Number, optional
    """
    def __init__(self, name: str = 'Network', R: float = 20.0) -> None:
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
        self.connections = []
        self.graph = Digraph(filename=(self.params['name']+'.gv'))

        # Compiler options
        self.params['spiking'] = False
        self.params['delay'] = False
        self.params['electrical'] = False
        self.params['electricalRectified'] = False
        self.params['gated'] = False
        self.params['numChannels'] = 0

    def get_num_neurons(self) -> int:
        """
        Calculate the number of neurons in the network, including within populations.

        :return: The number of neurons in the network.
        :rtype: int
        """
        num_neurons = 0
        for pop in self.populations:
            num_neurons += pop['number']
        return num_neurons

    def get_num_connections(self) -> int:
        """
        Calculate the number of connections in the network. Does not account for individual synapses between neurons in
        populations.

        :return: The number of connections in the network.
        :rtype: int
        """
        return len(self.connections)

    def get_num_populations(self) -> int:
        """
        Get the number of populations in the network.

        :return: Number of populations.
        :rtype: int
        """
        return len(self.populations)

    def get_num_inputs(self) -> int:
        """
        Get the number of input nodes to the network. Vector inputs are treated as a single input node.

        :return: Number of inputs.
        :rtype: int
        """
        return len(self.inputs)

    def get_num_inputs_actual(self) -> int:
        """
        Calculate the number of individual inputs throughout the network. Vector inputs have one input per dimension.

        :return: Number of individual inputs.
        :rtype: int
        """
        index = 0
        for inp in range(self.get_num_inputs()):
            size = self.inputs[inp]['size']
            index += size
        return index

    def get_num_outputs(self) -> int:
        """
        Get the number of output nodes from the network. Vector outputs have one output node.

        :return: Number of outputs.
        :rtype: int
        """
        return len(self.outputs)

    def get_num_outputs_actual(self) -> int:
        """
        Calculate the number of individual outputs from the network. Vector outputs have one output per dimension.

        :return: Number of individual outputs.
        :rtype: int
        """
        index = 0
        for out in range(self.get_num_outputs()):
            source_pop = self.outputs[out]['source']
            num_source_neurons = self.populations[source_pop]['number']
            # for num in range(num_source_neurons):
            #     index += 1
            index += num_source_neurons
        return index

    def get_population_index(self, name: str) -> int:
        """
        Given a string, find the numerical index of the population corresponding to that name within the network.

        :param name: Name of the population to find.
        :type name: str
        :return: Index of the population.
        :rtype: int
        """
        if not isinstance(name,str):
            raise TypeError('Name must be a valid string')
        for index in range(len(self.populations)):
            if self.populations[index]['name'] == name:
                return index
        raise ValueError('Population not found by name \'%s\'' % str(name))

    def get_connection_index(self, name: str) -> int:
        """
        Given a string, find the numerical index of the connection corresponding to that name within the network.

        :param name: Name of the connection to find.
        :type name: str
        :return: Index of the connection.
        :rtype: int
        """
        if not isinstance(name,str):
            raise TypeError('Name must be a valid string')
        for index in range(len(self.populations)):
            if self.connections[index]['name'] == name:
                return index
        raise ValueError('Population not found by name \'%s\'' % str(name))

    def get_input_index(self, name: str) -> int:
        """
        Given a string, find the numerical index of the input node given by that name within the network.

        :param name: Name of the input node to find.
        :type name: str
        :return: Index of the input node.
        :rtype: int
        """
        if not isinstance(name,str):
            raise TypeError('Name must be a valid string')
        for index in range(len(self.inputs)):
            if self.inputs[index]['name'] == name:
                return index
        raise ValueError('Input not found by name \'%s\'' % str(name))

    def render_graph(self, imgFormat: str = 'png', view: bool = False) -> None:
        """
        Render an image of the network in the form of a directed graph (DG) using graphviz.

        :param imgFormat: File extension of the resulting image, default is 'png'.
        :type imgFormat: str, optional
        :param view: Boolean flag to view the image, default is 'False'.
        :type view: bool, optional
        :return: None
        :rtype: N/A
        """
        if not isinstance(view,bool):
            raise TypeError('View must be a boolean')
        self.graph.format = imgFormat
        self.graph.render(view=view,cleanup=True)

    # Construction
    def add_population(self, neuron_type: Neuron, shape, name: str = None, color=None, initial_value=None) -> None:
        """
        Add a neural population to the network.

        :param neuron_type:  Type of neuron to add.
        :type neuron_type: sns_toolbox.design.neurons.Neuron
        :param shape: The number of neurons in each dimension of the population.
        :type shape: np.ndarray or torch.tensor
        :param name: Name of the population, default is the original name.
        :type name: str, optional
        :param color: Color of the population in the rendered image, default is the original color.
        :type color: str, optional
        :param initial_value: Initial value of membrane voltage for each neuron in the population. Must be either a
            single value, or an array matching 'shape'. Default value is the original value.
        :type initial_value: class: 'numbers.Number', class: 'np.ndarray', or class: 'torch.tensor'
        :return: None
        :rtype: N/A
        """
        if not isinstance(neuron_type, Neuron):
            raise TypeError('Input type is not a neuron')
        if len(shape) > 1: # population is multidimensional
            total_num_neurons = shape[0] * shape[1]
        else:
            total_num_neurons = shape[0]
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
        if initial_value is None:
            if total_num_neurons > 1:
                if neuron_type is SpikingNeuron:
                    initial_value = np.linspace(0,neuron_type.params['threshold_initial_value'],num=total_num_neurons)
                else:
                    initial_value = np.linspace(0,self.params['R'],num=total_num_neurons)
            else:
                initial_value = 0.0
        self.populations.append({'type': copy.deepcopy(neuron_type),
                                 'number': int(total_num_neurons),
                                 'shape': shape,
                                 'name': name,
                                 'color': color,
                                 'initial_value': initial_value})
        if total_num_neurons > 1:
            if len(shape) > 1:
                self.graph.node(str(len(self.populations) - 1), name,
                                style='filled, rounded',
                                shape='box3d',  # Populations with multiple neurons are marked with an outline
                                fillcolor=color,
                                fontcolor=font_color)
            else:
                self.graph.node(str(len(self.populations)-1), name,
                                style='filled, rounded',
                                shape='rect',   # Populations with multiple neurons are marked with an outline
                                fillcolor=color,
                                fontcolor=font_color)
        else:
            self.graph.node(str(len(self.populations) - 1), name,
                            style='filled',
                            fillcolor=color,
                            fontcolor=font_color)

        if neuron_type.params['spiking']:
            self.params['spiking'] = True

        if neuron_type.params['gated']:
            self.params['gated'] = True
            self.params['numChannels'] = max(self.params['numChannels'], neuron_type.params['numChannels'])

    def add_neuron(self, neuron_type: Neuron, name=None, color=None, initial_value=0.0) -> None:
        """
        Add a neuron to the network. Note that this is just a special case of addPopulation, which makes a
        population of 1 neuron.

        :param neuron_type:  Type of neuron to add.
        :type neuron_type: sns_toolbox.design.neurons.Neuron
        :param name:        Name of the neuron, defaults to the original name.
        :type name:         str, optional
        :param color:       Color of the neuron in the visual render, defaults to the original color.
        :type color:        str, optional
        :param initial_value: Initial value of the membrane voltage, defaults to 0.0. Units are millivolts (mV).
        :type initial_value: Number
        :return:    None
        :rtype: N/A
        """
        self.add_population(neuron_type, shape=[1], name=name, color=color, initial_value=initial_value)

    def add_input(self, dest: Any, size: int = 1, name: str = 'Input', color='white') -> None:
        """
        Add an input source to the network.

        :param dest:        Destination this input connects to.
        :type dest:         int or string
        :param size:        Number of input elements, default is 1.
        :type size:         int, optional
        :param name:        Name of the input node, default is 'Input'.
        :type name:         str, optional
        :param color:       Color of the input node in the visual render, default is 'white'.
        :type color:        str, optional
        :return:    None
        :rtype:     N/A
        """

        if not isinstance(name,str):
            raise TypeError('Name must be a string')
        if not valid_color(color):
            warnings.warn('Specified color is not in the standard SVG set. Defaulting to white.')
            color = 'white'
        if not isinstance(size,int):
            raise TypeError('Input size must be an integer greater than zero')
        if size <= 0:
            raise ValueError('Input size must be greater than 0')
        if not isinstance(dest, int):
            if isinstance(dest, str):
                dest = self.get_population_index(dest)
            else:
                raise TypeError('Destination index must be an integer or name')
        if dest > (len(self.populations)-1):
            raise ValueError('Destination index is out of range')
        if (size > 1) and (self.populations[dest]['number'] != size):
            raise ValueError('Input vector must be either size 1 or same size as destination population')
        font_color = set_text_color(color)
        self.inputs.append({'name': name,
                            'size': size,
                            'destination': dest,
                            'color': color})
        if size == 1:
            self.graph.node('In'+str(len(self.inputs) - 1), name,
                            style='filled',
                            shape='invhouse',
                            fillcolor=color,
                            fontcolor=font_color)
        else:
            self.graph.node('In' + str(len(self.inputs) - 1), name,
                            style='filled,diagonals',
                            shape='invhouse',
                            fillcolor=color,
                            fontcolor=font_color)
        self.graph.edge('In'+str(len(self.inputs) - 1), str(dest))

    def add_output(self, source: Any, name: str = 'Output', spiking: bool = False, color: str = 'white') -> None:
        """
        Add an output node to the network.

        :param source:      Source this output is connected to.
        :type source:       int or str
        :param name:        Name of the node, defaults to 'Output'.
        :type name:         str, optional
        :param spiking:     Flag for if this node stores voltage or spikes, default is 'False'.
        :type spiking:      bool, optional
        :param color:       Color of the output in the visual render, default is 'white'.
        :type color:        str, optional
        :return: None
        :rtype: N/A
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
                             'spiking': spiking,
                             'color': color})
        if spiking:
            if self.populations[source]['number'] == 1:
                self.graph.node('Out'+str(len(self.outputs) - 1), name,
                                style='filled',
                                shape='triangle',
                                fillcolor=color,
                                fontcolor=font_color)
            else:
                self.graph.node('Out'+str(len(self.outputs) - 1), name,
                                style='filled,diagonals',
                                shape='triangle',
                                fillcolor=color,
                                fontcolor=font_color)
        else:
            if self.populations[source]['number'] == 1:
                self.graph.node('Out' + str(len(self.outputs) - 1), name,
                                style='filled',
                                shape='house',
                                fillcolor=color,
                                fontcolor=font_color)
            else:
                self.graph.node('Out' + str(len(self.outputs) - 1), name,
                                style='filled,diagonals',
                                shape='house',
                                fillcolor=color,
                                fontcolor=font_color)

        self.graph.edge(str(source),'Out'+str(len(self.outputs)-1))

    def add_connection(self, connection_type: Connection, source: Any,
                       destination: Any, name: str = None, view_label: bool = False) -> None:
        """
        Add a synaptic connection between two populations in the network.

        :param connection_type: Type of connection to add.
        :type connection_type: sns_toolbox.design.connections.Connection
        :param source:      Index or name of source population in the network.
        :type source:       int or str
        :param destination: Index or name of destination population in the network.
        :type destination:  int or str
        :param name:        Name of synapse, default is the original name.
        :type name:         str, optional
        :param view_label:  Flag to render the name on the output graph, default is 'False'.
        :type view_label:   bool, optional
        :return: None
        :rtype: N/A
        """
        if not isinstance(connection_type, Connection):
            raise TypeError('Connection type must inherit from type Connection')
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
            label = connection_type.params['name']
        else:
            if isinstance(name,str):
                label = name
            else:
                raise TypeError('Name must be a string')
        self.connections.append({'name': label,
                                 'source': source,
                                 'destination': destination,
                                 'params': connection_type.params,
                                 'type': copy.deepcopy(connection_type),
                                 'view': view_label})

        if connection_type.params['pattern']:   # Pattern connection
            if self.populations[source]['number'] == 1:
                raise TypeError('Pattern connections are not supported for source populations of size 1')
            elif self.populations[source]['shape'] != self.populations[destination]['shape']:
                raise TypeError('Pattern connections are not currently supported for populations of different shape')
            style = 'vee'
            direction = 'forward'
            # 2d populations
            if len(self.populations[source]['shape']) > 1:
                g_max = __kernel_connections_2d__(self.populations[source]['shape'],
                                                  connection_type.params['max_conductance'])
                self.connections[-1]['params']['max_conductance'] = g_max
                del_e = __kernel_connections_2d__(self.populations[source]['shape'],
                                                  connection_type.params['relative_reversal_potential'])
                self.connections[-1]['params']['relative_reversal_potential'] = del_e
                if connection_type.params['spiking']:
                    time_constant = __kernel_connections_2d__(self.populations[source]['shape'],
                                                              connection_type.params['synapticTimeConstant'])
                    self.connections[-1]['params']['synapticTimeConstant'] = time_constant
                    transmit_delay = __kernel_connections_2d__(self.populations[source]['shape'],
                                                               connection_type.params['transmissionDelay'])
                    self.connections[-1]['params']['transmissionDelay'] = transmit_delay
            # 1d populations
            else:
                g_max = __kernel_connections_1d__(self.populations[source]['number'],
                                                  connection_type.params['max_conductance'])
                self.connections[-1]['params']['max_conductance'] = g_max
                del_e = __kernel_connections_1d__(self.populations[source]['number'],
                                                  connection_type.params['relative_reversal_potential'])
                self.connections[-1]['params']['relative_reversal_potential'] = del_e
                if connection_type.params['spiking']:
                    time_constant = __kernel_connections_1d__(self.populations[source]['number'],
                                                              connection_type.params['synapticTimeConstant'])
                    self.connections[-1]['params']['synapticTimeConstant'] = time_constant
                    transmit_delay = __kernel_connections_1d__(self.populations[source]['number'],
                                                               connection_type.params['transmissionDelay'])
                    self.connections[-1]['params']['transmissionDelay'] = transmit_delay
        elif connection_type.params['electrical']:
            style = 'odiamond'
            if connection_type.params['rectified'] is False:
                direction = 'both'
                self.params['electrical'] = True
            else:
                direction = 'forward'
                self.params['electricalRectified'] = True
        else:   # Chemical synapse
            direction = 'forward'
            if connection_type.params['relative_reversal_potential'] > 0:
                style = 'invempty'
            elif connection_type.params['relative_reversal_potential'] < 0:
                style = 'dot'
            else:
                style = 'odot'

        if view_label:
            self.graph.edge(str(source),
                            str(destination), dir=direction, arrowhead=style, label=label, arrowtail=style)
        else:
            self.graph.edge(str(source),
                            str(destination), dir=direction, arrowhead=style, arrowtail=style)

        if connection_type.params['spiking']:
            self.params['spiking'] = True
            if connection_type.params['transmissionDelay'] > 0:
                self.params['delay'] = True

    def add_network(self, network: 'Network', color: str = None) -> None:
        """
        Add an existing topology of inputs, outputs, and populations to the network.

        :param network: Network to copy over.
        :type network:  sns_toolbox.design.networks.Network
        :param color:   Color to render nodes in the network, default is the original colors.
        :type color:    str, optional
        :return: None
        :rtype: N/A
        """
        if not isinstance(network, Network):
            raise TypeError('Network must be of type Network')

        num_inputs = len(self.inputs)
        num_populations = len(self.populations)
        num_outputs = len(self.outputs)

        if color is None:
            for population in network.populations:
                self.add_population(neuron_type=population['type'], shape=population['shape'],
                                    name=population['name'], color=population['color'])
            for inp in network.inputs:
                self.add_input(dest=inp['destination'] + num_populations, name=inp['name'], color=inp['color']
                               ,size=inp['size'])
            for out in network.outputs:
                self.add_output(source=out['source'] + num_populations, name=out['name'], color=out['color'],
                                spiking=out['spiking'])
            for connection in network.connections:
                self.add_connection(connection_type=connection['type'], source=connection['source'] + num_populations,
                                    destination=connection['destination']+num_populations, view_label=connection['view'])
        else:
            if not valid_color(color):
                warnings.warn('Specified color is not in the standard SVG set. Defaulting to white.')
                color = 'white'
            for population in network.populations:
                self.add_population(neuron_type=population['type'], shape=population['shape'],
                                    name=population['name'], color=color)
            for inp in network.inputs:
                self.add_input(dest=inp['destination'] + num_populations, name=inp['name'], color=color
                               ,size=inp['size'])
            for out in network.outputs:
                self.add_output(source=out['source'] + num_populations, name=out['name'], color=color,
                                spiking=out['spiking'])
            for connection in network.connections:
                self.add_connection(connection_type=connection['type'], source=connection['source'] + num_populations,
                                    destination=connection['destination']+num_populations, view_label=connection['view'])

    def copy(self):
        """
        Create a copy of the network.

        :return: A new network with the same properties as the original. Each can be edited without effecting the other.
        :rtype: sns_toolbox.design.networks.Network
        """
        new_net = Network(name=self.params['name'],R=self.params['R'])
        new_net.add_network(self)

        return new_net

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
HELPER FUNCTIONS
"""

def __kernel_connections_1d__(pop_size,kernel):
    """
    Generate a connection matrix from a kernel vector and population size.

    :param pop_size: Number of neurons in the population.
    :type pop_size: int
    :param kernel: Kernel vector to apply.
    :type kernel: list, np.ndarray, or torch.tensor
    :return: Connection matrix.
    :rtype: np.ndarray
    """
    kernel_length = len(kernel)
    pad_amt = int((kernel_length-1)/2)
    connection_matrix = np.zeros([pop_size,pop_size])
    for row in range(pop_size):
        padded = np.zeros(pop_size + 2 * pad_amt)
        padded[row:row+kernel_length] = kernel
        connection_matrix[row,:] = padded[pad_amt:-pad_amt]
    return connection_matrix

def __kernel_connections_2d__(pop_shape,kernel):
    """
    Generate a connection matrix from a kernel matrix and population shape.

    :param pop_shape: Shape of the population.
    :type pop_shape: list, np.ndarray, or torch.tensor
    :param kernel: Kernel matrix to apply.
    :type kernel: np.ndarray or torch.Tensor
    :return: Connection matrix.
    :rtype: np.ndarray
    """
    kernel_rows = kernel.shape[0]
    kernel_cols = kernel.shape[1]
    num_kernel_dims = len(kernel.shape)
    pop_size = pop_shape[0]*pop_shape[1]
    pad_dims = []
    for dim in range(num_kernel_dims):
        pad_amt = int((kernel.shape[dim] - 1) / 2)
        pad_dims.append([pad_amt,pad_amt])
    source_matrix = np.zeros(pop_shape)
    connection_matrix = np.zeros([pop_size,pop_size])
    index = 0
    for row in range(pop_shape[0]):
        for col in range(pop_shape[1]):
            padded_matrix = np.pad(source_matrix, pad_dims)
            padded_matrix[row:row+kernel_rows,col:col+kernel_cols] = kernel
            pad_rows = pad_dims[0][0]
            pad_cols = pad_dims[1][0]
            if pad_cols == 0:
                subsection = padded_matrix[pad_rows:-pad_rows,:]
            elif pad_rows == 0:
                subsection = padded_matrix[:,pad_cols:-pad_cols]
            else:
                subsection = padded_matrix[pad_rows:-pad_rows,pad_cols:-pad_cols]
            connection_matrix[index,:] = subsection.flatten()
            index += 1
    return connection_matrix

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SPECIFIC MODELS
"""

class AdditionNetwork(Network):
    """
    Network which performs addition or subtraction of multiple inputs. Currently only supports non-spiking neurons.

    :param gains:       List of addition or subtraction weights.
    :type gains:        list, np.ndarray, or torch.tensor
    :param add_del_e:   Reversal potential of addition synapses, default is 100. Unit is millivolts (mV).
    :type add_del_e:    Number, optional
    :param sub_del_e:   Reversal potential of subtraction synapses, default is -40. Unit is millivolts (mV).
    :type sub_del_e:    Number, optional
    :param neuron_type: Neuron preset to use, default is sns_toolbox.design.neurons.NonSpikingNeuron.
    :type neuron_type:  sns_toolbox.design.neurons.NonSpikingNeuron, optional
    :param name:        Name of this network, default is 'Add'.
    :type name:         str, optional
    """
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
            self.add_connection(conn, i + 1, name + 'Sum')

class MultiplicationNetwork(Network):
    """
    Network where the activity in neuron 0 is roughly multiplied by the activity in neuron 1. Currently only supports
    non-spiking neurons.

    :param neuron_type: Neuron preset to use, default is sns_toolbox.design.neurons.NonSpikingNeuron.
    :type neuron_type:  sns_toolbox.design.neurons.NonSpikingNeuron, optional
    :param name:        Name of this network, default is 'Multiply'.
    :type name:         str, optional
    """
    def __init__(self,neuron_type=NonSpikingNeuron(),name='Multiply',**kwargs):
        super().__init__(name=name,**kwargs)
        self.add_neuron(neuron_type, name=name + '0')
        self.add_neuron(neuron_type, name=name + '1')
        self.add_neuron(neuron_type, name=name + 'Inter')
        self.add_neuron(neuron_type, name=name + 'Result')

        transmit = NonSpikingTransmissionSynapse(gain=1.0,R=self.params['R'])
        conductance = -self.params['R']/-1.0
        modulate_special = NonSpikingSynapse(max_conductance=conductance, relative_reversal_potential=-1.0)

        self.add_connection(transmit, name + '0', name + 'Result')
        self.add_connection(modulate_special, name + '1', name + 'Inter')
        self.add_connection(modulate_special, name + 'Inter', name + 'Result')

class DivisionNetwork(Network):
    """
    Network where the activity in neuron 0 is roughly divided by the activity in neuron 1. Currently only supports
    non-spiking neurons.

    :param neuron_type: Neuron preset to use, default is sns_toolbox.design.neurons.NonSpikingNeuron.
    :type neuron_type:  sns_toolbox.design.neurons.NonSpikingNeuron, optional
    :param ratio:       Modulation ratio.
    :type ratio:        Number
    :param name:        Name of this network, default is 'Divide'.
    :type name:         str, optional
    """
    def __init__(self,gain,ratio,name='Divide',neuron_type=NonSpikingNeuron(),**kwargs):
        super().__init__(name=name,**kwargs)
        self.add_neuron(neuron_type, name=name + 'Transmit')
        self.add_neuron(neuron_type, name=name + 'Modulate')
        self.add_neuron(neuron_type, name=name + 'Results')

        transmission = NonSpikingTransmissionSynapse(gain,R=self.params['R'])
        self.add_connection(transmission, 0, 2)

        modulation = NonSpikingModulationSynapse(ratio)
        self.add_connection(modulation, 1, 2)

class DifferentiatorNetwork(Network):
    """
    Network where the activity of 'Uout' is the derivative of the activity in 'Uin'. Currently only supports
    non-spiking neurons.

    :param slew_rate:   Steepest signal this differentiator can handle, default is 1.0. Influences the difference in
        internal time constants.
    :type slew_rate:    Number, optional
    :param name:        Name of this network, default is 'Differentiate'.
    :type name:         str, optional
    :param tau_fast:    Time constant of the faster neurons, default is 1.0. Units are milliseconds (ms).
    :type tau_fast:     Number, optional
    """
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

        self.add_connection(add_synapse, 'Uin', 'Ufast')
        self.add_connection(add_synapse, 'Uin', 'Uslow')
        self.add_connection(add_synapse, 'Ufast', 'Uout')
        self.add_connection(sub_synapse, 'Uslow', 'Uout')

class IntegratorNetwork(Network):
    """
    Network where the activity of 'Uint' integrates the applied stimulus to itself over time. Currently only supports
    non-spiking neurons.

    :param integration_gain:            Gain of the integration, default is 1.0.
    :type integration_gain:             Number, optional
    :param relative_reversal_potential: Reversal potential of the synapses in the network, default is -40.0. Must be
        less than 0. Units are millivolts (mV).
    :type relative_reversal_potential:  Number, optional
    :param name:                        Name of this network, default is 'Integrator'.
    :type name:                         str, optional
    """
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

        self.add_connection(synapse_type, 0, 1)
        self.add_connection(synapse_type, 1, 0)


# class AdaptationNetwork(Network):
#     def __init__(self,ratio=0.5,name='Adaptation',neuron_type=NonSpikingNeuron(),**kwargs):
#         super().__init__(name=name,**kwargs)
#         relative_reversal_potential = 2*(self.params['R']**2 + 1)/self.params['R']
#         conductance_fast_slow = self.params['R']/(relative_reversal_potential - self.params['R'])
#         conductance_slow_fast = (self.params['R']*(ratio-1)*(relative_reversal_potential - self.params['R'] +
#                                                              ratio*self.params['R']))/(ratio *
#                                                                                        relative_reversal_potential *
#                                                                                        (-relative_reversal_potential -
#                                                                                         ratio*self.params['R']))
#         fast_slow = NonSpikingSynapse(max_conductance=conductance_fast_slow,
#                                       relative_reversal_potential=relative_reversal_potential)
#         slow_fast = NonSpikingSynapse(max_conductance=conductance_slow_fast,
#                                       relative_reversal_potential=-relative_reversal_potential)
#
#         self.add_neuron(neuron_type,name='Uadapt')
#         self.add_neuron(neuron_type)
#
#         self.add_connection(fast_slow, 0, 1)
#         self.add_connection(slow_fast, 1, 0)