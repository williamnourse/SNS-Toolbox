Building Networks
"""""""""""""""""

The primary activity when using SNS-Toolbox is building networks. We do this using :code:`Network` objects.

Creating a Network
===================

To create a network, we initialize a :code:`Network`
object
::
    from sns_toolbox.networks import Network

    net = Network(name='Network')

Adding to a Network
====================

Once a :code:`Network` has been created, we can fill it with the various neurons and connections that we desire.

Adding a Neuron
----------------

To add a neuron to the network, we first define the properties of the neuron and then it is added to the network. We'll
add a basic :code:`NonSpikingNeuron` here, for a full list of all available neural models please see `Neural Models
<https://sns-toolbox.readthedocs.io/en/latest/neuron_models.html>`_.
::
    from sns_toolbox.neurons import NonSpikingNeuron # There are other types available, this is the most basic

    neuron_type = NonSpikingNeuron()
    net.add_neuron(neuron_type, name='Source Nrn')

Adding a Population of Neurons
-------------------------------

Instead of adding a single neuron, we can add an entire population of neurons. The process is much the same as adding a
single neuron:
::
    net.add_population(neuron_type, [2,3], name='Dest Pop', initial_value=None)

The first parameter is the type of neuron to add, and the second parameter is list describing the shape of
the population. Currently neural populations of 1-2 dimensions are supported. If no initial value is provided for the
population, the neural resting potential is assumed.

Adding Inputs
--------------

In order to add external influence to a network, we need to add an input. This is done by specifying the neuron which
the input is injected into, either via the numeric index of the neuron or its name.
::
    net.add_input('Source Nrn', name='Input')
    # net.add_input(0, name='Input')  # This is equally valid

If the input is going into a population, the input can be either a single element or a vector. For a vector input, the
size must be specified. The following line adds a 4 element vector input to the neural population.
::
    net.add_input('Dest Pop', size=4, name='Pop Input')

Currently inputs can only be a single element, or a one-dimensional vector.

Adding Outputs
---------------

Adding output nodes to the network allows for desired neural states to be accessed outside the network. To add an
output, the target neuron or population must be specified, as well as whether the desired output state is the neuron's
membrane voltage or spiking state.
::
    net.add_output('Source Nrn', name='Output 0', spiking=False)
    net.add_output('Dest Pop', name='Output 1', spiking=False)

When an output node is added to a neural population, one individual output node is generated for each neuron in the
population. So as an example, an output node added to population with 6 neurons will generate 6 output nodes.

Adding Connections
-------------------

Connections can be added between neurons/populations within a network. A connection type is required, along with the
source neuron/population and the destination (these can be strings or integer indices).
::
    synapse_type = NonSpikingSynapse()
    net.add_connection(synapse_type, 'Source Nrn', 'Dest Pop')

When connecting to or from populations with multiple neurons, traditional synapses or pattern connections can be used.
If a synapse is used, the synaptic conductance is evenly split across all-to-all synapses between the neurons. Pattern
connections explicitly specify a pattern of synaptic connectivity, however they are limited to connecting populations of
the same size and shape. For a full listing of available connection types, please consult
`Connection Models <https://sns-toolbox.readthedocs.io/en/latest/connection_models.html>`_.

Adding Networks
----------------

Existing :code:`Network` objects can also be incorporated into any other network. For the process please see
`Tutorial 4 <https://sns-toolbox.readthedocs.io/en/latest/tutorials/tutorial_4.html>`_, and for all of the networks
provided by SNS-Toolbox please see
`sns_toolbox.networks <https://sns-toolbox.readthedocs.io/en/latest/autoapi/sns_toolbox/networks/index.html>`_.

Other Functionality
====================

Copying a Network
------------------

Networks can be copied using their native method.
::
    net_copy = net.copy()

Compiling a Network
--------------------

Networks can be compiled to a software backend for efficient simulation. For an overview of this process, please consult
`Compiling and Simulating Networks <https://sns-toolbox.readthedocs.io/en/latest/compiling.html>`_.

Getting Network Properties
---------------------------

:code:`Network` objects have a variety of internal methods which return network properties. These are the following:
::
    # Calculate the number of neurons in the network, including within populations.
    num_neurons = net.get_num_neurons()

    # Calculate the number of connection objects in the network
    num_connections = net.get_num_connections()

    # Get the number of populations in the network
    num_pop = net.get_num_populations()

    # Get the number of input nodes to the network. Vector inputs are treated as a single input node.
    num_inputs =  net.get_num_inputs()

    # Calculate the number of individual inputs throughout the network. Vector inputs have one input per dimension.
    num_inputs_actual = net.get_num_inputs_actual() -> int:

    # Get the number of output nodes from the network. Population outputs have one output node.
    num_outputs = net.get_num_outputs()

    # Calculate the number of individual outputs from the network. Population outputs have one output per neuron.
    num_outputs_actual = net.get_num_outputs_actual()

    # Given a string, find the numerical index of the population corresponding to that name within the network.
    index_pop = net.get_population_index('name')

    # Given a string, find the numerical index of the connection corresponding to that name within the network.
    index_conn get_connection_index('name')

    # Given a string, find the numerical index of the input node given by that name within the network.
    index_input = net.get_input_index('name')