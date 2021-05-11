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

import __utilities__
import neurons
import synapses
import networks

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
            with self.assertRaises(ValueError):
                testNeuron = neurons.Neuron(name=1)
        with self.subTest():
            with self.assertRaises(ValueError):
                testNeuron = neurons.Neuron(membraneCapacitance='foo')
        with self.subTest():
            with self.assertRaises(ValueError):
                testNeuron = neurons.Neuron(membraneConductance='foo')
        with self.subTest():
            with self.assertRaises(ValueError):
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
            with self.assertRaises(ValueError):
                testNeuron = neurons.NonSpikingNeuron(name=1)
        with self.subTest():
            with self.assertRaises(ValueError):
                testNeuron = neurons.NonSpikingNeuron(membraneCapacitance='foo')
        with self.subTest():
            with self.assertRaises(ValueError):
                testNeuron = neurons.NonSpikingNeuron(membraneConductance='foo')
        with self.subTest():
            with self.assertRaises(ValueError):
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
        with self.assertRaises(ValueError):
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
            with self.assertRaises(ValueError):
                testSynapse = synapses.NonSpikingSynapse(name=5)
        with self.subTest():
            with self.assertRaises(ValueError):
                testSynapse = synapses.NonSpikingSynapse(maxConductance='foo')
        with self.subTest():
            with self.assertRaises(ValueError):
                testSynapse = synapses.NonSpikingSynapse(relativeReversalPotential='foo')
        with self.subTest():
            with self.assertRaises(ValueError):
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
            with self.assertRaises(ValueError): # not a number
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



"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MAIN TESTING FUNCTION
"""

if __name__ == '__main__':
    unittest.main()

from neurons import NonSpikingNeuron
from synapses import NonSpikingSynapse
from networks import NonSpikingNetwork

# simple = NonSpikingNeuron()
# slow = NonSpikingNeuron(membraneCapacitance=50)
# transmit = NonSpikingSynapse(name='Transmit')
# inhibit = NonSpikingSynapse(name='Inhibit', relativeReversalPotential=-40)
# modulate = NonSpikingSynapse(name='Mod', relativeReversalPotential=0)
#
# smallNet = NonSpikingNetwork(name='SmallNet')
# smallNet.addNeuron(simple,suffix='A',color='blue')
# smallNet.addNeuron(simple,suffix='B')
# smallNet.addNeuron(slow,color='orange')
# smallNet.addSynapse(transmit,0,2,viewLabel=True)
# smallNet.addSynapse(transmit,1,2,viewLabel=True)
# smallNet.renderGraph(view=True)
#
# bigNet = NonSpikingNetwork(name='BigNet')
# bigNet.addNeuron(simple,suffix='Origin')
# bigNet.addNeuron(simple,suffix='Modulate',color='indianred')
# bigNet.addNetwork(smallNet,color='teal')
# bigNet.addSynapse(inhibit,0,1,viewLabel=True)
# bigNet.addSynapse(modulate,1,2,viewLabel=True)
# bigNet.renderGraph(view=True)
