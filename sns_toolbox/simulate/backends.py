"""
Simulation backends for nonspiking networks. Each of these are python-based, and are constructed using a Nonspiking
Network. They can then be run for a step, with the inputs being a vector of neural states and applied currents and the
output being the next step of neural states.
William Nourse
August 31, 2021
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
import sys

from sns_toolbox.design.networks import Network
from sns_toolbox.design.neurons import NonSpikingNeuron, SpikingNeuron
from sns_toolbox.design.connections import SpikingSynapse, NonSpikingSynapse

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
BASE CLASS
"""

class Backend:
    """
    Base-level class for all simulation backends. Each will do the following:
        - Construct a representation of a given network using the desired backend technique
        - Take in (some form of) a vector of input states and applied currents, and compute the result for the next
          timestep
    """
    def __init__(self, network: Network, dt: float = 0.1) -> None:
        """
        Construct the backend based on the network design
        :param network: NonSpikingNetwork to serve as a design template
        :param dt:      Simulation time constant
        """
        self.dt = dt
        self.numPopulations = network.getNumPopulations()
        self.numNeurons = network.getNumNeurons()
        self.numSynapses = network.getNumSynapses()
        self.numInputs = network.getNumInputs()
        self.numOutputs = network.getNumOutputs()
        self.R = network.params['R']

    def forward(self, inputs) -> Any:
        """
        Compute the next neural states based on previous neural states
        :param inputs:    Input currents into the network
        :return:          The next neural voltages
        """
        raise NotImplementedError

"""
########################################################################################################################
NUMPY BACKEND

Simulating the network using numpy vectors and matrices. Note that this is not sparse, so memory may explode for large networks
"""

class SNS_Numpy(Backend):
    def __init__(self,network: Network, **kwargs):
        super().__init__(network,**kwargs)

        """Neurons"""
        # Initialize the vectors
        self.U = np.zeros(self.numNeurons)
        self.Ulast = np.zeros(self.numNeurons)
        self.spikes = np.zeros(self.numNeurons)
        Cm = np.zeros(self.numNeurons)
        self.Gm = np.zeros(self.numNeurons)
        self.Ib = np.zeros(self.numNeurons)
        self.theta0 = np.zeros(self.numNeurons)
        self.theta = np.zeros(self.numNeurons)
        self.thetaLast = np.zeros(self.numNeurons)
        self.m = np.zeros(self.numNeurons)
        tauTheta = np.zeros(self.numNeurons)

        # iterate over the populations in the network
        popsAndNrns = []
        index = 0
        for pop in range(len(network.populations)):
            numNeurons = network.populations[pop]['number'] # find the number of neurons in the population
            popsAndNrns.append([])
            Ulast = 0.0
            for num in range(numNeurons):   # for each neuron, copy the parameters over
                Cm[index] = network.populations[pop]['type'].params['membraneCapacitance']
                self.Gm[index] = network.populations[pop]['type'].params['membraneConductance']
                self.Ib[index] = network.populations[pop]['type'].params['bias']
                self.Ulast[index] = Ulast
                if isinstance(network.populations[pop]['type'],SpikingNeuron):  # if the neuron is spiking, copy more
                    self.theta0[index] = network.populations[pop]['type'].params['thresholdInitialValue']
                    Ulast += network.populations[pop]['type'].params['thresholdInitialValue']/numNeurons
                    self.m[index] = network.populations[pop]['type'].params['thresholdProportionalityConstant']
                    tauTheta[index] = network.populations[pop]['type'].params['thresholdTimeConstant']
                else:   # otherwise, set to the special values for NonSpiking
                    self.theta0[index] = sys.float_info.max
                    self.m[index] = 0
                    tauTheta[index] = 1
                    Ulast += self.R/numNeurons
                popsAndNrns[pop].append(index)
                index += 1
        self.U = np.copy(self.Ulast)
        # set the derived vectors
        self.timeFactorMembrane = self.dt/Cm
        self.timeFactorThreshold = self.dt/tauTheta
        self.theta = np.copy(self.theta0)
        self.thetaLast = np.copy(self.theta0)

        """Inputs"""
        self.inputConnectivity = np.zeros([self.numNeurons, self.numInputs])  # initialize connectivity matrix
        for conn in network.inputConns:  # iterate over the connections in the network
            wt = conn['weight']  # get the weight
            source = conn['source']  # get the source
            destPop = conn['destination']  # get the destination
            for dest in popsAndNrns[destPop]:
                self.inputConnectivity[dest][source] = wt  # set the weight in the correct source and destination

        """Synapses"""
        # initialize the matrices
        self.GmaxNon = np.zeros([self.numNeurons, self.numNeurons])
        self.GmaxSpk = np.zeros([self.numNeurons, self.numNeurons])
        self.Gspike = np.zeros([self.numNeurons, self.numNeurons])
        self.DelE = np.zeros([self.numNeurons, self.numNeurons])
        self.tauSyn = np.zeros([self.numNeurons, self.numNeurons])+1

        # iterate over the synapses in the network
        for syn in range(len(network.synapses)):
            sourcePop = network.synapses[syn]['source']
            destPop = network.synapses[syn]['destination']
            Gmax = network.synapses[syn]['type'].params['maxConductance']
            delE = network.synapses[syn]['type'].params['relativeReversalPotential']

            if isinstance(network.synapses[syn]['type'],SpikingSynapse):
                tauS = network.synapses[syn]['type'].params['synapticTimeConstant']
                for source in popsAndNrns[sourcePop]:
                    for dest in popsAndNrns[destPop]:
                        self.GmaxSpk[dest][source] = Gmax/len(popsAndNrns[sourcePop])
                        self.DelE[dest][source] = delE
                        self.tauSyn[dest][source] = tauS
            else:
                for source in popsAndNrns[sourcePop]:
                    for dest in popsAndNrns[destPop]:
                        self.GmaxNon[dest][source] = Gmax/len(popsAndNrns[sourcePop])
                        self.DelE[dest][source] = delE
        self.timeFactorSynapse = self.dt/self.tauSyn

        """Outputs"""
        # Figure out how many outputs there actually are, since an output has as many elements as its input population
        outputs = []
        index = 0
        for out in range(len(network.outputs)):
            sourcePop = network.outputs[out]['source']
            numSourceNeurons = network.populations[sourcePop]['number']
            outputs.append([])
            for num in range(numSourceNeurons):
                outputs[out].append(index)
                index += 1
        self.numOutputs = index

        self.outputVoltageConnectivity = np.zeros([self.numOutputs, self.numNeurons])  # initialize connectivity matrix
        self.outputSpikeConnectivity = np.copy(self.outputVoltageConnectivity)
        for out in range(len(network.outputs)):  # iterate over the connections in the network
            wt = network.outputs[out]['weight']  # get the weight
            sourcePop = network.outputs[out]['source']  # get the source
            for i in range(len(popsAndNrns[sourcePop])):
                if network.outputs[out]['spiking']:
                    self.outputSpikeConnectivity[outputs[out][i]][popsAndNrns[sourcePop][i]] = wt  # set the weight in the correct source and destination
                else:
                    self.outputVoltageConnectivity[outputs[out][i]][popsAndNrns[sourcePop][i]] = wt  # set the weight in the correct source and destination
        print('Input Connectivity:')
        print(self.inputConnectivity)
        print('GmaxNon:')
        print(self.GmaxNon)
        print('GmaxSpike:')
        print(self.GmaxSpk)
        print('DelE:')
        print(self.DelE)
        print('Output Voltage Connectivity')
        print(self.outputVoltageConnectivity)
        print('Output Spike Connectivity:')
        print(self.outputSpikeConnectivity)

    def forward(self, inputs) -> Any:
        self.Ulast = np.copy(self.U)
        self.thetaLast = np.copy(self.theta)
        Iapp = np.matmul(self.inputConnectivity, inputs)  # Apply external current sources to their destinations
        Gnon = np.maximum(0, np.minimum(self.GmaxNon * self.Ulast / self.R, self.GmaxNon))
        self.Gspike = self.Gspike * (1 - self.timeFactorSynapse)
        Gsyn = Gnon + self.Gspike
        Isyn = np.sum(Gsyn * self.DelE, axis=1) - self.Ulast * np.sum(Gsyn, axis=1)
        self.U = self.Ulast + self.timeFactorMembrane * (-self.Gm * self.Ulast + self.Ib + Isyn + Iapp)  # Update membrane potential
        self.theta = self.thetaLast + self.timeFactorThreshold * (-self.thetaLast + self.theta0 + self.m * self.Ulast)  # Update the firing thresholds
        self.spikes = np.sign(np.minimum(0, self.theta - self.U))  # Compute which neurons have spiked
        self.Gspike = np.maximum(self.Gspike, (-self.spikes) * self.GmaxSpk)  # Update the conductance of synapses which spiked
        self.U = self.U * (self.spikes + 1)  # Reset the membrane voltages of neurons which spiked

        return np.matmul(self.outputVoltageConnectivity, self.U) + np.matmul(self.outputSpikeConnectivity, -self.spikes)

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MANUAL BACKEND
"""

class SNS_Manual(Backend):
    """
    This is the most straightforward (smooth-brain) approach to simulating a system of non-spiking neurons. Each
    Diff Eq and Synapses is evaluated one-by-one at each timestep. Simplest to keep track of, but (probably) quite
    slow at scale. Note that this method stores a dense matrix of synaptic parameters, which has problematic scaling
    for memory requirements!
    """
    def __init__(self, network: Network, **kwargs) -> None:
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

class SNS_SciPy(Backend):
    """
    This is the simplest approach to simulating large networks using sparse matrices. Whether it scales to methods run
    on GPUs remains to be seen, but should definitely use less memory than the naive, manual approach.
    """
    def __init__(self, network: Network, **kwargs) -> None:
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

class SNS_Pytorch(Backend):
    def __init__(self, network: Network, device: torch.device = None, **kwargs):
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
