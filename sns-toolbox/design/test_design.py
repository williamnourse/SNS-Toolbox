"""
Test the code within the design suite
William Nourse
May 11, 2021
Why do I have to be Mr. Pink
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



"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SYNAPSES TESTS
"""



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
