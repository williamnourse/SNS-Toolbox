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
import numpy as np
import torch
from scipy.sparse import csr_matrix, lil_matrix

from sns_toolbox.design.networks import NonSpikingNetwork

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
BASE CLASS
"""

class NonSpikingBackend:
    """
    Base-level class for all simulation backends. Each will do the following:
        - Construct a representation of a given network using the desired backend technique
        - Take in (some form of) a vector of input states and applied currents, and compute the result for the next
          timestep
    """
    def __init__(self, network: NonSpikingNetwork, dt: float = 0.1) -> None:
        """
        Construct the backend based on the network design
        :param network: NonSpikingNetwork to serve as a design template
        :param dt:      Simulation time constant
        """
        self.dt = dt
        self.numNeurons = network.getNumNeurons()
        self.numSynapses = network.getNumSynapses()
        self.R = network.params['R']

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

class SNS_Manual(NonSpikingBackend):
    """
    This is the most straightforward (smooth-brain) approach to simulating a system of non-spiking neurons. Each
    Diff Eq and Synapses is evaluated one-by-one at each timestep. Simplest to keep track of, but (probably) quite
    slow at scale. Note that this method stores a dense matrix of synaptic parameters, which has problematic scaling
    for memory requirements!
    """
    def __init__(self, network: NonSpikingNetwork,**kwargs) -> None:
        """
        Initialize the backend

        :param network: NonspikingNetwork to serve as a design template
        :param kwargs:  Other parameters passed to the base class (dt)
        """
        super().__init__(network,**kwargs)
        # Network Parameters
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
        """
        Compute the next neural states
        :param statesLast:      Neural states at the last timestep
        :param appliedCurrents: External currents at the current timestep
        :return:                Neural states at the current timestep
        """
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

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SCIPY
"""

class SNS_SciPy(NonSpikingBackend):
    """
    This is the simplest approach to simulating large networks using sparse matrices. Whether it scales to methods run
    on GPUs remains to be seen, but should definitely use less memory than the naive, manual approach.
    """
    def __init__(self, network: NonSpikingNetwork,**kwargs) -> None:
        """
        Initialize the backend
        :param network: NonSpikingNetwork to serve as a design template
        :param kwargs:  Parameters passed to the base class (dt)
        """
        super().__init__(network,**kwargs)

        # Neural Parameters
        Cm = np.zeros(self.numNeurons)
        Gm = np.zeros(self.numNeurons)
        Ibias = np.zeros(self.numNeurons)

        for i in range(self.numNeurons):
            Cm[i] = network.neurons[i].params['membraneCapacitance']
            Gm[i] = network.neurons[i].params['membraneConductance']
            Ibias[i] = network.neurons[i].params['bias']

        GmRow = np.array(list(range(self.numNeurons)))
        self.Ibias = csr_matrix(Ibias)
        self.GmArr = csr_matrix((Gm,(GmRow, GmRow)), shape=(self.numNeurons, self.numNeurons))
        self.timeFactor = csr_matrix(self.dt/Cm)

        # Synaptic Parameters
        sources = []
        destinations = []
        gMaxVals = []
        delEVals = []
        for i in range(self.numSynapses):
            sources.append(network.synapses[i].params['source'])
            destinations.append(network.synapses[i].params['destination'])
            gMaxVals.append(network.synapses[i].params['maxConductance'])
            delEVals.append(network.synapses[i].params['relativeReversalPotential'])

        self.gMax = csr_matrix((gMaxVals,(destinations,sources)),shape=(self.numNeurons,self.numNeurons))
        self.delE = csr_matrix((delEVals, (destinations, sources)), shape=(self.numNeurons, self.numNeurons))

    def forward(self, statesLast: csr_matrix, appliedCurrents: csr_matrix) -> csr_matrix:
        """
        Compute the next neural states
        :param statesLast:      Neural states at the last timestep
        :param appliedCurrents: External currents at the current timestep
        :return:                Neural states at the current timestep
        """
        # Compute the synaptic conductance matrix
        Gsyn = self.gMax.minimum(self.gMax.multiply(statesLast/self.R))
        Gsyn = Gsyn.maximum(0)

        # Compute the following for each neuron:
        # Isyn[j] = sum(elementwise(G[:,j], delE[:,j])) - Ulast[j]*sum(G[:,j])
        Isyn = lil_matrix(np.zeros(self.numNeurons))
        for i in range(self.numNeurons):
            Isyn[0, i] = (Gsyn[i, :].multiply(self.delE[i, :])).sum() - statesLast[0, i] * Gsyn[i, :].sum()

        # Compute the next state
        statesNext = statesLast + self.timeFactor.multiply(-(statesLast @ self.GmArr) + self.Ibias + Isyn + appliedCurrents)

        return statesNext

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
PYTORCH SPARSE
"""

class SNS_Pytorch(NonSpikingBackend):
    def __init__(self, network: NonSpikingNetwork, device: torch.device = None,**kwargs):
        super().__init__(network,**kwargs)

        # Neural parameters
        Cm = np.zeros(self.numNeurons)
        Gm = np.zeros(self.numNeurons)
        Ibias = np.zeros(self.numNeurons)

        for i in range(self.numNeurons):
            Cm[i] = network.neurons[i].params['membraneCapacitance']
            Gm[i] = network.neurons[i].params['membraneConductance']
            Ibias[i] = network.neurons[i].params['bias']

        GmRow = np.array(list(range(self.numNeurons)))
        self.Ibias = torch.as_tensor(Ibias)#.to_sparse()
        self.GmArr = torch.sparse_coo_tensor([GmRow,GmRow],Gm,[self.numNeurons,self.numNeurons])
        self.timeFactor = torch.as_tensor(self.dt / Cm)#.to_sparse()

        # Synapse parameters
        sources = []
        destinations = []
        gMaxVals = []
        delEVals = []
        for i in range(self.numSynapses):
            sources.append(network.synapses[i].params['source'])
            destinations.append(network.synapses[i].params['destination'])
            gMaxVals.append(network.synapses[i].params['maxConductance'])
            delEVals.append(network.synapses[i].params['relativeReversalPotential'])

        self.gMax = torch.sparse_coo_tensor([destinations,sources],gMaxVals,[self.numNeurons,self.numNeurons])
        self.delE = torch.sparse_coo_tensor([destinations,sources],delEVals,[self.numNeurons,self.numNeurons])
