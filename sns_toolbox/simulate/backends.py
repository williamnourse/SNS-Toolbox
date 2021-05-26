"""
Simulation backends for nonspiking networks. Each of these are python-based, and are constructed using a Nonspiking
Network. They can then be run for a step, with the inputs being a vector of neural states and applied currents and the
output being the next step of neural states.
William Nourse
May 26, 2021
I've heard that you're a low-down Yankee liar
"""

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
IMPORTS
"""

from typing import Any
from numbers import Number
import numpy as np

from sns_toolbox.design.networks import NonSpikingNetwork

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
BASE CLASS
"""

class NonSpikingBackend:
    def __init__(self, network: NonSpikingNetwork, dt: float = 0.1) -> None:
        """
        Construct the backend based on the network design
        :param network: NonSpikingNetwork to serve as a design template
        """
        self.dt = dt
        self.numNeurons = network.getNumNeurons()
        self.numSynapses = network.getNumSynapses()

    def forward(self, statesLast: Any, appliedCurrents: Any) -> Any:
        """
        Compute the next neural states based on previous neural states
        :param statesLast:    Input neural voltages (states)
        :param appliedCurrents: External applied currents at the current step
        :return:                The next neural voltages
        """
        raise NotImplementedError

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MANUAL BACKEND
"""

class Manual(NonSpikingBackend):
    def __init__(self, network: NonSpikingNetwork,**kwargs):
        super().__init__(network,**kwargs)
        # Network Parameters
        self.R = network.params['R']
        self.statesNext = np.zeros(self.numNeurons)

        # Neural Parameters
        self.Cm = np.zeros(self.numNeurons)
        self.Gm = np.zeros(self.numNeurons)
        self.Ibias = np.zeros(self.numNeurons)

        for i in range(self.numNeurons):
            self.Cm[i] = network.neurons[i].params['membraneCapacitance']
            self.Gm[i] = network.neurons[i].params['membraneConductance']
            self.Ibias[i] = network.neurons[i].params['bias']

        # Synaptic Parameters
        self.gMax = np.zeros([self.numNeurons, self.numNeurons])
        self.delE = np.zeros([self.numNeurons, self.numNeurons])
        self.Gsyn = np.zeros([self.numNeurons,self.numNeurons])

        for i in range(self.numSynapses):
            source = network.synapses[i].params['source']
            dest = network.synapses[i].params['destination']
            self.gMax[dest,source] = network.synapses[i].params['maxConductance']
            self.delE[dest,source] = network.synapses[i].params['relativeReversalPotential']

    def forward(self, statesLast: np.ndarray, appliedCurrents: np.ndarray) -> np.ndarray:
        for source in range(self.numNeurons):
            for dest in range(self.numNeurons):
                self.Gsyn[dest,source] = self.gMax[dest,source] * max(0, min(1, statesLast[source] / self.R))
                #print('Made it')

        for i in range(self.numNeurons):
            Isyn = 0
            for pre in range(self.numNeurons):
                Isyn += (self.Gsyn[i,pre] * (self.delE[i,pre] - statesLast[i]))

            self.statesNext[i] = statesLast[i] + self.dt/self.Cm[i] * (-self.Gm[i]*statesLast[i] + self.Ibias[i] + Isyn + appliedCurrents[i])

        return self.statesNext

