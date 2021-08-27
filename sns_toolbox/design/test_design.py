"""
Test the code within the design suite
William Nourse
May 11, 2021
Why do I have to be Mr. Pink?
"""

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
IMPORTS
"""

import unittest
import warnings

from sns_toolbox.design import __utilities__
from sns_toolbox.design import neurons
from sns_toolbox.design import synapses
from sns_toolbox.design import networks

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
UTILITIES TESTS
"""

class TestValidColor(unittest.TestCase):
    def test_color_in_set(self):
        self.assertEqual(True, __utilities__.validColor('khaki'), 'Should be True')

    def test_color_not_in_set(self):
        self.assertEqual(False, __utilities__.validColor('not a color'), 'Should be False')

    def test_input_not_string(self):
        self.assertEqual(False, __utilities__.validColor(5), 'Should be False')

    def test_input_none(self):
        self.assertEqual(False, __utilities__.validColor(), 'Should be False')

    def test_color_in_set_uppercase(self):
        self.assertEqual(False, __utilities__.validColor('ROYALblue'), 'Should be False')

    def test_color_tuple(self):
        self.assertEqual(False, __utilities__.validColor(['navy','olive']))


class TestSetTextColor(unittest.TestCase):
    def test_color_with_white_text(self):
        self.assertEqual('white', __utilities__.setTextColor('brown'), 'Should be white')

    def test_color_with_black_text(self):
        self.assertEqual('black', __utilities__.setTextColor('white'), 'Should be black')

    def test_uppercase_white_text(self):
        self.assertEqual('black', __utilities__.setTextColor('Brown'), 'Should be black')

    def test_not_string(self):
        warnings.simplefilter('ignore',category=UserWarning)
        self.assertEqual('black', __utilities__.setTextColor(5), 'Should be black')

    def test_color_tuple(self):
        warnings.simplefilter('ignore', category=UserWarning)
        self.assertEqual('black', __utilities__.setTextColor(['black','blue']), 'Should be black')

    def test_input_none(self):
        self.assertEqual('black', __utilities__.setTextColor(), 'Should be black')

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
NEURONS TESTS
"""

class TestNeuron(unittest.TestCase):
    def test_construct_default(self):
        testNeuron = neurons.Neuron()
        with self.subTest():    # Name
            self.assertEqual(testNeuron.params['name'],'Neuron','Should be Neuron')
        with self.subTest():    # Color
            self.assertEqual(testNeuron.params['color'], 'white', 'Should be white')
        with self.subTest():    # Membrane Capacitance
            self.assertEqual(testNeuron.params['membraneCapacitance'],5.0,'Should be 5.0')
        with self.subTest():    # Membrane Conductance
            self.assertEqual(testNeuron.params['fontColor'],'black','Should be black')
        with self.subTest():    # Bias current
            self.assertEqual(testNeuron.params['bias'],0.0,'Should be 0.0')

    def test_construct_valid(self):
        testNeuron = neurons.Neuron(name='Name',
                                    color='blue',
                                    membraneCapacitance=2.0,
                                    membraneConductance=2.0,
                                    bias=2.0)
        with self.subTest():  # Name
            self.assertEqual(testNeuron.params['name'], 'Name', 'Should be Name')
        with self.subTest():  # Color
            self.assertEqual(testNeuron.params['color'], 'blue', 'Should be blue')
        with self.subTest():  # Membrane Capacitance
            self.assertEqual(testNeuron.params['membraneCapacitance'], 2.0, 'Should be 2.0')
        with self.subTest():  # Membrane Conductance
            self.assertEqual(testNeuron.params['fontColor'], 'white', 'Should be white')
        with self.subTest():  # Bias current
            self.assertEqual(testNeuron.params['bias'], 2.0, 'Should be 2.0')

    def test_construct_invalid(self):
        with self.subTest():
            with self.assertRaises(TypeError):
                testNeuron = neurons.Neuron(name=1)
        with self.subTest():
            with self.assertRaises(TypeError):
                testNeuron = neurons.Neuron(membraneCapacitance='foo')
        with self.subTest():
            with self.assertRaises(TypeError):
                testNeuron = neurons.Neuron(membraneConductance='foo')
        with self.subTest():
            with self.assertRaises(TypeError):
                testNeuron = neurons.Neuron(bias='foo')


class TestNonSpikingNeuron(unittest.TestCase):
    def test_construct_default(self):
        testNeuron = neurons.NonSpikingNeuron()
        with self.subTest():    # Name
            self.assertEqual(testNeuron.params['name'],'Neuron','Should be Neuron')
        with self.subTest():    # Color
            self.assertEqual(testNeuron.params['color'], 'white', 'Should be white')
        with self.subTest():    # Membrane Capacitance
            self.assertEqual(testNeuron.params['membraneCapacitance'],5.0,'Should be 5.0')
        with self.subTest():    # Membrane Conductance
            self.assertEqual(testNeuron.params['fontColor'],'black','Should be black')
        with self.subTest():    # Bias current
            self.assertEqual(testNeuron.params['bias'],0.0,'Should be 0.0')

    def test_construct_valid(self):
        testNeuron = neurons.NonSpikingNeuron(name='Name',
                                              color='blue',
                                              membraneCapacitance=2.0,
                                              membraneConductance=2.0,
                                              bias=2.0)
        with self.subTest():  # Name
            self.assertEqual(testNeuron.params['name'], 'Name', 'Should be Name')
        with self.subTest():  # Color
            self.assertEqual(testNeuron.params['color'], 'blue', 'Should be blue')
        with self.subTest():  # Membrane Capacitance
            self.assertEqual(testNeuron.params['membraneCapacitance'], 2.0, 'Should be 2.0')
        with self.subTest():  # Membrane Conductance
            self.assertEqual(testNeuron.params['fontColor'], 'white', 'Should be white')
        with self.subTest():  # Bias current
            self.assertEqual(testNeuron.params['bias'], 2.0, 'Should be 2.0')

    def test_construct_invalid(self):
        with self.subTest():
            with self.assertRaises(TypeError):
                testNeuron = neurons.NonSpikingNeuron(name=1)
        with self.subTest():
            with self.assertRaises(TypeError):
                testNeuron = neurons.NonSpikingNeuron(membraneCapacitance='foo')
        with self.subTest():
            with self.assertRaises(TypeError):
                testNeuron = neurons.NonSpikingNeuron(membraneConductance='foo')
        with self.subTest():
            with self.assertRaises(TypeError):
                testNeuron = neurons.NonSpikingNeuron(bias='foo')

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SYNAPSES TESTS
"""

class TestSynapse(unittest.TestCase):
    def test_construct_default(self):
        testSynapse = synapses.Synapse()
        self.assertEqual(testSynapse.params['name'],'Synapse','Should be Synapse')

    def test_construct_valid(self):
        testSynapse = synapses.Synapse(name='Name')
        self.assertEqual(testSynapse.params['name'],'Name','Should be Name')

    def test_construct_invalid(self):
        with self.assertRaises(TypeError):
            testSynapse = synapses.Synapse(name=5)


class TestNonSpikingSynapse(unittest.TestCase):
    def test_construct_default(self):
        testSynapse = synapses.NonSpikingSynapse()
        with self.subTest():
            self.assertEqual(testSynapse.params['name'],'Synapse','Should be Synapse')
        with self.subTest():
            self.assertEqual(testSynapse.params['maxConductance'],1.0,'Should be 1.0')
        with self.subTest():
            self.assertEqual(testSynapse.params['relativeReversalPotential'],40.0,'Should be 40.0')
        with self.subTest():
            self.assertEqual(testSynapse.params['R'],20.0,'Should be 20.0')

    def test_construct_valid(self):
        testSynapse = synapses.NonSpikingSynapse(name='Name',maxConductance=2.0,relativeReversalPotential=2.0,R=2.0)
        with self.subTest():
            self.assertEqual(testSynapse.params['name'], 'Name', 'Should be Name')
        with self.subTest():
            self.assertEqual(testSynapse.params['maxConductance'], 2.0, 'Should be 2.0')
        with self.subTest():
            self.assertEqual(testSynapse.params['relativeReversalPotential'], 2.0, 'Should be 2.0')
        with self.subTest():
            self.assertEqual(testSynapse.params['R'], 2.0, 'Should be 2.0')

    def test_construct_invalid(self):
        with self.subTest():
            with self.assertRaises(TypeError):
                testSynapse = synapses.NonSpikingSynapse(name=5)
        with self.subTest():
            with self.assertRaises(TypeError):
                testSynapse = synapses.NonSpikingSynapse(maxConductance='foo')
        with self.subTest():
            with self.assertRaises(TypeError):
                testSynapse = synapses.NonSpikingSynapse(relativeReversalPotential='foo')
        with self.subTest():
            with self.assertRaises(TypeError):
                testSynapse = synapses.NonSpikingSynapse(R='foo')
        with self.subTest():
            with self.assertRaises(ValueError):
                testSynapse = synapses.NonSpikingSynapse(maxConductance=0)
        with self.subTest():
            with self.assertRaises(ValueError):
                testSynapse = synapses.NonSpikingSynapse(R=0)
        with self.subTest():
            with self.assertRaises(ValueError):
                testSynapse = synapses.NonSpikingSynapse(maxConductance=-1)
        with self.subTest():
            with self.assertRaises(ValueError):
                testSynapse = synapses.NonSpikingSynapse(R=-1)


class TestTransmissionSynapse(unittest.TestCase):
    def test_construct_default(self):
        testSynapse = synapses.TransmissionSynapse()
        with self.subTest():
            self.assertEqual(((1.0*20.0)/(40.0-1.0*20.0)),testSynapse.params['maxConductance'],'Should be 1.0')
        with self.subTest():
            self.assertEqual('Transmit',testSynapse.params['name'],'Should be Transmit')

    def test_construct_valid(self):
        testSynapse = synapses.TransmissionSynapse(gain=1.5,name='Name')
        with self.subTest():
            self.assertEqual(((1.5 * 20.0) / (40.0 - 1.5 * 20.0)), testSynapse.params['maxConductance'],
                             'Should be 3.0')
        with self.subTest():
            self.assertEqual('Name', testSynapse.params['name'], 'Should be Name')

    def test_construct_invalid(self):
        with self.subTest():
            with self.assertRaises(TypeError): # not a number
                testSynapse = synapses.TransmissionSynapse(gain='foo')
        with self.subTest():
            with self.assertRaises(ValueError): # = 0
                testSynapse = synapses.TransmissionSynapse(gain=0)
        with self.subTest():
            with self.assertRaises(ValueError):  # < 0
                testSynapse = synapses.TransmissionSynapse(gain=-1)
        with self.subTest():
            with self.assertRaises(ValueError): # gain causes gMax < 0
                testSynapse = synapses.TransmissionSynapse(gain=3.0)


class TestModulationSynapse(unittest.TestCase):
    def test_construct_default(self):
        testSynapse = synapses.ModulationSynapse()
        with self.subTest():
            self.assertEqual(0.0,testSynapse.params['relativeReversalPotential'],'Should be 0.0')
        with self.subTest():
            self.assertEqual('Modulate',testSynapse.params['name'],'Should be Modulate')


"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
NETWORKS TESTS
"""

class TestNonSpikingNetwork(unittest.TestCase):
    # I'm going to ignore anything with graphviz, that's better debugged visually
    def test_construct_default(self):
        testNetwork = networks.Network()
        with self.subTest():
            self.assertEqual(testNetwork.params['name'],'Network','Should be Network')
        with self.subTest():
            self.assertEqual(testNetwork.neurons,[],'Should be empty list')
        with self.subTest():
            self.assertEqual(testNetwork.synapses,[],'Should be empty list')

    def test_construct_valid(self):
        testNetwork = networks.Network(name='Name')
        with self.subTest():
            self.assertEqual(testNetwork.params['name'], 'Name', 'Should be Name')
        with self.subTest():
            self.assertEqual(testNetwork.neurons, [], 'Should be empty list')
        with self.subTest():
            self.assertEqual(testNetwork.synapses, [], 'Should be empty list')

    def test_construct_invalid(self):
        with self.assertRaises(TypeError):
            testNetwork = networks.Network(name=5)

    def test_getNumNeurons_empty(self):
        testNetwork = networks.Network()
        self.assertEqual(testNetwork.getNumNeurons(),0,'Should be 0')

    def test_getNumNeurons_not_empty(self):
        testNetwork = networks.Network()
        testNetwork.neurons = [1,2]
        self.assertEqual(testNetwork.getNumNeurons(),2,'Should be 2')

    def test_getNumSynapses_empty(self):
        testNetwork = networks.Network()
        self.assertEqual(testNetwork.getNumSynapses(),0,'Should be 0')

    def test_getNumSynapses_not_empty(self):
        testNetwork = networks.Network()
        testNetwork.synapses = [1,2]
        self.assertEqual(testNetwork.getNumSynapses(),2,'Should be 2')

    def test_addNeuron_default(self):
        testNetwork = networks.Network()
        testNeuron = neurons.NonSpikingNeuron() # Defaults: name-Neuron, color-white, memCap-5.0, memCond-1.0, bias-0.0
        testNetwork.addNeuron(testNeuron)
        with self.subTest():
            self.assertEqual(testNetwork.neurons[0].params['name'],'Neuron','Should be Neuron')
        with self.subTest():
            self.assertEqual(testNetwork.neurons[0].params['color'],'white', 'Should be white')
        with self.subTest():
            self.assertEqual(testNetwork.neurons[0].params['fontColor'],'black', 'Should be black')

    def test_addNeuron_copy(self):
        testNetwork = networks.Network()
        testNeuron = neurons.NonSpikingNeuron()  # Defaults: name-Neuron, color-white, memCap-5.0, memCond-1.0, bias-0.0
        testNetwork.addNeuron(testNeuron)
        self.assertNotEqual(testNeuron,testNetwork.neurons[0])

    def test_addNeuron_valid(self):
        testNetwork = networks.Network()
        testNeuron = neurons.NonSpikingNeuron()  # Defaults: name-Neuron, color-white, memCap-5.0, memCond-1.0, bias-0.0
        testNetwork.addNeuron(testNeuron,suffix='Test',color='indianred')
        with self.subTest():
            self.assertEqual(testNetwork.neurons[0].params['name'],'NeuronTest', 'Should be NeuronTest')
        with self.subTest():
            self.assertEqual(testNetwork.neurons[0].params['color'],'indianred','Should be indianred')

    def test_addNeuron_invalid(self):
        testNetwork = networks.Network()
        testNeuron = neurons.NonSpikingNeuron()  # Defaults: name-Neuron, color-white, memCap-5.0, memCond-1.0, bias-0.0
        with self.subTest():
            with self.assertRaises(TypeError):
                testNetwork.addNeuron(testNeuron,suffix=5)
        with self.subTest():
            with self.assertRaises(TypeError):
                testNetwork.addNeuron(5)

    def test_addSynapse_default(self):
        testNetwork = networks.Network()
        testNeuron = neurons.NonSpikingNeuron()  # Defaults: name-Neuron, color-white, memCap-5.0, memCond-1.0, bias-0.0
        testNetwork.addNeuron(testNeuron)
        testNetwork.addNeuron(testNeuron,suffix='2')
        testSynapse = synapses.NonSpikingSynapse()
        testNetwork.addSynapse(testSynapse,0,1)
        with self.subTest():
            self.assertNotEqual(testNetwork.synapses[0],testSynapse)
        with self.subTest():
            self.assertEqual(testNetwork.synapses[0].params['source'],0,'Should be 0')
        with self.subTest():
            self.assertEqual(testNetwork.synapses[0].params['destination'], 1, 'Should be 1')
        with self.subTest():
            self.assertEqual(testNetwork.synapses[0].params['label'],None,'Should be None')

    def test_addSynapse_valid(self):
        testNetwork = networks.Network()
        testNeuron = neurons.NonSpikingNeuron()  # Defaults: name-Neuron, color-white, memCap-5.0, memCond-1.0, bias-0.0
        testNetwork.addNeuron(testNeuron)
        testNetwork.addNeuron(testNeuron, suffix='2')
        testSynapse = synapses.NonSpikingSynapse()
        testNetwork.addSynapse(testSynapse, 0, 0,viewLabel=True,offset=1)
        with self.subTest():
            self.assertEqual(testNetwork.synapses[0].params['source'],1,'Should be 1')
        with self.subTest():
            self.assertEqual(testNetwork.synapses[0].params['destination'], 1, 'Should be 2')
        with self.subTest():
            self.assertEqual(testNetwork.synapses[0].params['label'],'Synapse','Should be Synapse')

    def test_addSynapse_invalid(self):
        testNetwork = networks.Network()
        testNeuron = neurons.NonSpikingNeuron()  # Defaults: name-Neuron, color-white, memCap-5.0, memCond-1.0, bias-0.0
        testNetwork.addNeuron(testNeuron)
        testNetwork.addNeuron(testNeuron, suffix='2')
        testSynapse = synapses.NonSpikingSynapse()
        with self.subTest():
            with self.assertRaises(TypeError):
                testNetwork.addSynapse(10,0,1)
        with self.subTest():
            with self.assertRaises(TypeError):
                testNetwork.addSynapse(testSynapse,'foo',1)
        with self.subTest():
            with self.assertRaises(TypeError):
                testNetwork.addSynapse(testSynapse,0,'bar')
        with self.subTest():
            with self.assertRaises(TypeError):
                testNetwork.addSynapse(testSynapse,0,1,offset=2.2)
        with self.subTest():
            with self.assertRaises(TypeError):
                testNetwork.addSynapse(testSynapse,0,1,viewLabel='bad')
        with self.subTest():
            with self.assertRaises(ValueError):
                testNetwork.addSynapse(testSynapse,10,1)
        with self.subTest():
            with self.assertRaises(ValueError):
                testNetwork.addSynapse(testSynapse,-1,1)
        with self.subTest():
            with self.assertRaises(ValueError):
                testNetwork.addSynapse(testSynapse,0,10)
        with self.subTest():
            with self.assertRaises(ValueError):
                testNetwork.addSynapse(testSynapse,0,-1)
        with self.subTest():
            with self.assertRaises(ValueError):
                testNetwork.addSynapse(testSynapse,0,1,offset=10)
        with self.subTest():
            with self.assertRaises(ValueError):
                testNetwork.addSynapse(testSynapse,0,1,offset=1)

    def test_addNetwork_default(self):
        testNetwork = networks.Network()
        sourceNetwork = networks.Network()
        testNeuron = neurons.NonSpikingNeuron()  # Defaults: name-Neuron, color-white, memCap-5.0, memCond-1.0, bias-0.0
        sourceNetwork.addNeuron(testNeuron)
        sourceNetwork.addNeuron(testNeuron)
        testNetwork.addNeuron(testNeuron)
        testSynapse = synapses.NonSpikingSynapse()
        sourceNetwork.addSynapse(testSynapse, 0, 1,viewLabel=True)
        testNetwork.addNetwork(sourceNetwork)
        with self.subTest():
            self.assertEqual(len(testNetwork.neurons),3,'Should be 3')
        with self.subTest():
            self.assertEqual(len(testNetwork.synapses),1,'Should be 1')
        with self.subTest():
            self.assertEqual(testNetwork.neurons[0].params['color'],'white','Should be white')
        with self.subTest():
            self.assertEqual(testNetwork.neurons[1].params['color'],'white','Should be white')
        with self.subTest():
            self.assertEqual(testNetwork.neurons[2].params['color'],'white','Should be white')
        with self.subTest():
            self.assertEqual(testNetwork.synapses[0].params['source'],1,'Should be 1')
        with self.subTest():
            self.assertEqual(testNetwork.synapses[0].params['destination'],2,'Should be 2')
        with self.subTest():
            self.assertNotEqual(testNetwork.synapses[0].params['label'],None,'Should be Synapse')

    def test_addNetwork_valid(self):
        testNetwork = networks.Network()
        sourceNetwork = networks.Network()
        testNeuron = neurons.NonSpikingNeuron()  # Defaults: name-Neuron, color-white, memCap-5.0, memCond-1.0, bias-0.0
        sourceNetwork.addNeuron(testNeuron)
        sourceNetwork.addNeuron(testNeuron)
        testNetwork.addNeuron(testNeuron)
        testNetwork.addNetwork(sourceNetwork,color='navy')
        with self.subTest():
            self.assertEqual(testNetwork.neurons[0].params['color'],'white','Should be white')
        with self.subTest():
            self.assertEqual(testNetwork.neurons[1].params['color'],'navy','Should be navy')
        with self.subTest():
            self.assertEqual(testNetwork.neurons[2].params['color'],'navy','Should be navy')

    def test_addNetwork_invalid(self):
        testNetwork = networks.Network()
        with self.assertRaises(TypeError):
            testNetwork.addNetwork('foo')

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MAIN TESTING FUNCTION
"""

if __name__ == '__main__':
    unittest.main()
