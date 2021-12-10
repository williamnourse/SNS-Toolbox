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
import sys

from sns_toolbox.design.networks import Network
from sns_toolbox.design.neurons import SpikingNeuron
from sns_toolbox.design.connections import SpikingSynapse
from sns_toolbox.simulate.__utilities__ import sendVars

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
    def __init__(self, network: Network, dt: float = 0.1, debug: bool = False) -> None:
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
        self.debug = debug

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

Simulating the network using numpy vectors and matrices.
Note that this is not sparse, so memory may explode for large networks
"""

class SNS_Numpy(Backend):
    def __init__(self,network: Network,**kwargs):
        super().__init__(network,**kwargs)

        """Neurons"""
        if self.debug:
            print('BUILDING NEURONS')
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
        if self.debug:
            print('BUILDING INPUTS')
        self.inputConnectivity = np.zeros([self.numNeurons, self.numInputs])  # initialize connectivity matrix
        self.in_offset = np.zeros(self.numInputs)
        self.in_linear = np.zeros(self.numInputs)
        self.in_quad = np.zeros(self.numInputs)
        self.in_cubic = np.zeros(self.numInputs)
        self.inputs_mapped = np.zeros(self.numInputs)
        for inp in range(network.getNumInputs()):  # iterate over the connections in the network
            self.in_offset[inp] = network.inputs[inp]['offset']
            self.in_linear[inp] = network.inputs[inp]['linear']
            self.in_quad[inp] = network.inputs[inp]['quadratic']
            self.in_cubic[inp] = network.inputs[inp]['cubic']
            destPop = network.inputs[inp]['destination']  # get the destination
            for dest in popsAndNrns[destPop]:
                self.inputConnectivity[dest][inp] = 1.0  # set the weight in the correct source and destination

        """Synapses"""
        if self.debug:
            print('BUILDING SYNAPSES')
        # initialize the matrices
        self.GmaxNon = np.zeros([self.numNeurons, self.numNeurons])
        self.GmaxSpk = np.zeros([self.numNeurons, self.numNeurons])
        self.Gspike = np.zeros([self.numNeurons, self.numNeurons])
        self.DelE = np.zeros([self.numNeurons, self.numNeurons])
        self.tauSyn = np.zeros([self.numNeurons, self.numNeurons])+1
        spikeDelays = np.zeros([self.numNeurons,self.numNeurons])
        self.spikeRows = []
        self.spikeCols = []
        self.bufferSteps = []
        self.bufferNrns = []
        self.spikeDelayInds = np.zeros([self.numNeurons**2])
        self.delayedSpikes = np.zeros([self.numNeurons,self.numNeurons])

        # iterate over the synapses in the network
        for syn in range(len(network.synapses)):
            sourcePop = network.synapses[syn]['source']
            destPop = network.synapses[syn]['destination']
            Gmax = network.synapses[syn]['type'].params['maxConductance']
            delE = network.synapses[syn]['type'].params['relativeReversalPotential']

            if isinstance(network.synapses[syn]['type'],SpikingSynapse):
                tauS = network.synapses[syn]['type'].params['synapticTimeConstant']
                delay = network.synapses[syn]['type'].params['synapticTransmissionDelay']
                for source in popsAndNrns[sourcePop]:
                    for dest in popsAndNrns[destPop]:
                        self.GmaxSpk[dest][source] = Gmax/len(popsAndNrns[sourcePop])
                        self.DelE[dest][source] = delE
                        self.tauSyn[dest][source] = tauS
                        spikeDelays[dest][source] = delay
                        self.bufferNrns.append(source)
                        self.bufferSteps.append(delay)
                        self.spikeRows.append(dest)
                        self.spikeCols.append(source)
            else:
                for source in popsAndNrns[sourcePop]:
                    for dest in popsAndNrns[destPop]:
                        self.GmaxNon[dest][source] = Gmax/len(popsAndNrns[sourcePop])
                        self.DelE[dest][source] = delE
        self.timeFactorSynapse = self.dt/self.tauSyn

        bufferLength = int(np.max(spikeDelays)+1)
        # self.spikeDelayInds = spikeDelays.flatten()
        # self.spikeDelayInds = self.spikeDelayInds.astype(int)
        # index = 0
        # for row in range(self.numNeurons):
        #     for col in range(self.numNeurons):
        #         self.spikeRows[index] = row
        #         self.spikeCols[index] = col
        #         index += 1
        self.spikeBuffer = np.zeros([bufferLength, self.numNeurons])
        # self.spikeRows = self.spikeRows.astype(int)
        # self.spikeCols = self.spikeCols.astype(int)

        """Outputs"""
        if self.debug:
            print('BUILDING OUTPUTS')
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
        self.out_offset = np.zeros(self.numOutputs)
        self.out_linear = np.zeros(self.numOutputs)
        self.out_quad = np.zeros(self.numOutputs)
        self.out_cubic = np.zeros(self.numOutputs)
        self.outputs_raw = np.zeros(self.numOutputs)
        for out in range(len(network.outputs)):  # iterate over the connections in the network
            sourcePop = network.outputs[out]['source']  # get the source
            if network.outputs[out]['spiking']:
                self.out_linear[out] = 1.0
            else:
                self.out_offset[out] = network.outputs[out]['offset']
                self.out_linear[out] = network.outputs[out]['linear']
                self.out_quad[out] = network.outputs[out]['quadratic']
                self.out_cubic[out] = network.outputs[out]['cubic']
            for i in range(len(popsAndNrns[sourcePop])):
                if network.outputs[out]['spiking']:
                    self.outputSpikeConnectivity[outputs[out][i]][popsAndNrns[sourcePop][i]] = 1.0  # set the weight in the correct source and destination
                    self.out_linear[outputs[out][i]] = 1.0
                else:
                    self.outputVoltageConnectivity[outputs[out][i]][popsAndNrns[sourcePop][i]] = 1.0  # set the weight in the correct source and destination
                    self.out_offset[outputs[out][i]] = network.outputs[out]['offset']
                    self.out_linear[outputs[out][i]] = network.outputs[out]['linear']
                    self.out_quad[outputs[out][i]] = network.outputs[out]['quadratic']
                    self.out_cubic[outputs[out][i]] = network.outputs[out]['cubic']
        if self.debug:
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
            print('U:')
            print(self.U)
            print('Ulast:')
            print(self.Ulast)
            print('theta0:')
            print(self.theta0)
            print('ThetaLast:')
            print(self.thetaLast)
            print('Theta')
            print(self.theta)
            print('\nDONE BUILDING')

    def forward(self, inputs) -> Any:
        self.Ulast = np.copy(self.U)
        self.thetaLast = np.copy(self.theta)
        self.inputs_mapped = self.in_cubic*(inputs**3) + self.in_quad*(inputs**2) + self.in_linear*inputs + self.in_offset
        Iapp = np.matmul(self.inputConnectivity, self.inputs_mapped)  # Apply external current sources to their destinations
        Gnon = np.maximum(0, np.minimum(self.GmaxNon * self.Ulast / self.R, self.GmaxNon))
        self.Gspike = self.Gspike * (1 - self.timeFactorSynapse)
        Gsyn = Gnon + self.Gspike
        Isyn = np.sum(Gsyn * self.DelE, axis=1) - self.Ulast * np.sum(Gsyn, axis=1)
        self.U = self.Ulast + self.timeFactorMembrane * (-self.Gm * self.Ulast + self.Ib + Isyn + Iapp)  # Update membrane potential
        self.theta = self.thetaLast + self.timeFactorThreshold * (-self.thetaLast + self.theta0 + self.m * self.Ulast)  # Update the firing thresholds
        self.spikes = np.sign(np.minimum(0, self.theta - self.U))  # Compute which neurons have spiked

        # New stuff with delay
        self.spikeBuffer = np.roll(self.spikeBuffer,1,axis=0)   # Shift buffer entries down
        self.spikeBuffer[0,:] = self.spikes    # Replace row 0 with the current spike data
        # Update a matrix with all of the appropriately delayed spike values
        self.delayedSpikes[self.spikeRows,self.spikeCols] = self.spikeBuffer[self.bufferSteps,self.bufferNrns]

        self.Gspike = np.maximum(self.Gspike, (-self.delayedSpikes) * self.GmaxSpk)  # Update the conductance of synapses which spiked
        self.U = self.U * (self.spikes + 1)  # Reset the membrane voltages of neurons which spiked
        self.outputs_raw = np.matmul(self.outputVoltageConnectivity, self.U) + np.matmul(self.outputSpikeConnectivity, -self.spikes)

        return self.out_cubic*(self.outputs_raw**3) + self.out_quad*(self.outputs_raw**2)\
            + self.out_linear*self.outputs_raw + self.out_offset

class SNS_Numpy_No_Delay(Backend):
    def __init__(self,network: Network,**kwargs):
        super().__init__(network,**kwargs)

        """Neurons"""
        if self.debug:
            print('BUILDING NEURONS')
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
        if self.debug:
            print('BUILDING INPUTS')
        self.inputConnectivity = np.zeros([self.numNeurons, self.numInputs])  # initialize connectivity matrix
        self.in_offset = np.zeros(self.numInputs)
        self.in_linear = np.zeros(self.numInputs)
        self.in_quad = np.zeros(self.numInputs)
        self.in_cubic = np.zeros(self.numInputs)
        self.inputs_mapped = np.zeros(self.numInputs)
        for inp in range(network.getNumInputs()):  # iterate over the connections in the network
            self.in_offset[inp] = network.inputs[inp]['offset']
            self.in_linear[inp] = network.inputs[inp]['linear']
            self.in_quad[inp] = network.inputs[inp]['quadratic']
            self.in_cubic[inp] = network.inputs[inp]['cubic']
            destPop = network.inputs[inp]['destination']  # get the destination
            for dest in popsAndNrns[destPop]:
                self.inputConnectivity[dest][inp] = 1.0  # set the weight in the correct source and destination

        """Synapses"""
        if self.debug:
            print('BUILDING SYNAPSES')
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
        if self.debug:
            print('BUILDING OUTPUTS')
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
        self.out_offset = np.zeros(self.numOutputs)
        self.out_linear = np.zeros(self.numOutputs)
        self.out_quad = np.zeros(self.numOutputs)
        self.out_cubic = np.zeros(self.numOutputs)
        self.outputs_raw = np.zeros(self.numOutputs)
        for out in range(len(network.outputs)):  # iterate over the connections in the network
            sourcePop = network.outputs[out]['source']  # get the source
            if network.outputs[out]['spiking']:
                self.out_linear[out] = 1.0
            else:
                self.out_offset[out] = network.outputs[out]['offset']
                self.out_linear[out] = network.outputs[out]['linear']
                self.out_quad[out] = network.outputs[out]['quadratic']
                self.out_cubic[out] = network.outputs[out]['cubic']
            for i in range(len(popsAndNrns[sourcePop])):
                if network.outputs[out]['spiking']:
                    self.outputSpikeConnectivity[outputs[out][i]][popsAndNrns[sourcePop][i]] = 1.0  # set the weight in the correct source and destination
                    self.out_linear[outputs[out][i]] = 1.0
                else:
                    self.outputVoltageConnectivity[outputs[out][i]][popsAndNrns[sourcePop][i]] = 1.0  # set the weight in the correct source and destination
                    self.out_offset[outputs[out][i]] = network.outputs[out]['offset']
                    self.out_linear[outputs[out][i]] = network.outputs[out]['linear']
                    self.out_quad[outputs[out][i]] = network.outputs[out]['quadratic']
                    self.out_cubic[outputs[out][i]] = network.outputs[out]['cubic']
        if self.debug:
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
            print('U:')
            print(self.U)
            print('Ulast:')
            print(self.Ulast)
            print('theta0:')
            print(self.theta0)
            print('ThetaLast:')
            print(self.thetaLast)
            print('Theta')
            print(self.theta)
            print('\nDONE BUILDING')

    def forward(self, inputs) -> Any:
        self.Ulast = np.copy(self.U)
        self.thetaLast = np.copy(self.theta)
        self.inputs_mapped = self.in_cubic*(inputs**3) + self.in_quad*(inputs**2) + self.in_linear*inputs + self.in_offset
        Iapp = np.matmul(self.inputConnectivity, self.inputs_mapped)  # Apply external current sources to their destinations
        Gnon = np.maximum(0, np.minimum(self.GmaxNon * self.Ulast / self.R, self.GmaxNon))
        self.Gspike = self.Gspike * (1 - self.timeFactorSynapse)
        Gsyn = Gnon + self.Gspike
        Isyn = np.sum(Gsyn * self.DelE, axis=1) - self.Ulast * np.sum(Gsyn, axis=1)
        self.U = self.Ulast + self.timeFactorMembrane * (-self.Gm * self.Ulast + self.Ib + Isyn + Iapp)  # Update membrane potential
        self.theta = self.thetaLast + self.timeFactorThreshold * (-self.thetaLast + self.theta0 + self.m * self.Ulast)  # Update the firing thresholds
        self.spikes = np.sign(np.minimum(0, self.theta - self.U))  # Compute which neurons have spiked
        self.Gspike = np.maximum(self.Gspike, (-self.spikes) * self.GmaxSpk)  # Update the conductance of synapses which spiked
        self.U = self.U * (self.spikes + 1)  # Reset the membrane voltages of neurons which spiked
        self.outputs_raw = np.matmul(self.outputVoltageConnectivity, self.U) + np.matmul(self.outputSpikeConnectivity, -self.spikes)

        return self.out_cubic*(self.outputs_raw**3) + self.out_quad*(self.outputs_raw**2)\
            + self.out_linear*self.outputs_raw + self.out_offset

class SNS_Numpy_Non_Spiking(Backend):
    def __init__(self,network: Network,**kwargs):
        super().__init__(network,**kwargs)

        """Neurons"""
        if self.debug:
            print('BUILDING NEURONS')
        # Initialize the vectors
        self.U = np.zeros(self.numNeurons)
        self.Ulast = np.zeros(self.numNeurons)
        Cm = np.zeros(self.numNeurons)
        self.Gm = np.zeros(self.numNeurons)
        self.Ib = np.zeros(self.numNeurons)

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
                popsAndNrns[pop].append(index)
                index += 1
        self.U = np.copy(self.Ulast)
        # set the derived vectors
        self.timeFactorMembrane = self.dt/Cm

        """Inputs"""
        if self.debug:
            print('BUILDING INPUTS')
        self.inputConnectivity = np.zeros([self.numNeurons, self.numInputs])  # initialize connectivity matrix
        self.in_offset = np.zeros(self.numInputs)
        self.in_linear = np.zeros(self.numInputs)
        self.in_quad = np.zeros(self.numInputs)
        self.in_cubic = np.zeros(self.numInputs)
        self.inputs_mapped = np.zeros(self.numInputs)
        for inp in range(network.getNumInputs()):  # iterate over the connections in the network
            self.in_offset[inp] = network.inputs[inp]['offset']
            self.in_linear[inp] = network.inputs[inp]['linear']
            self.in_quad[inp] = network.inputs[inp]['quadratic']
            self.in_cubic[inp] = network.inputs[inp]['cubic']
            destPop = network.inputs[inp]['destination']  # get the destination
            for dest in popsAndNrns[destPop]:
                self.inputConnectivity[dest][inp] = 1.0  # set the weight in the correct source and destination

        """Synapses"""
        if self.debug:
            print('BUILDING SYNAPSES')
        # initialize the matrices
        self.GmaxNon = np.zeros([self.numNeurons, self.numNeurons])
        self.DelE = np.zeros([self.numNeurons, self.numNeurons])

        # iterate over the synapses in the network
        for syn in range(len(network.synapses)):
            sourcePop = network.synapses[syn]['source']
            destPop = network.synapses[syn]['destination']
            Gmax = network.synapses[syn]['type'].params['maxConductance']
            delE = network.synapses[syn]['type'].params['relativeReversalPotential']

            for source in popsAndNrns[sourcePop]:
                for dest in popsAndNrns[destPop]:
                    self.GmaxNon[dest][source] = Gmax/len(popsAndNrns[sourcePop])
                    self.DelE[dest][source] = delE

        """Outputs"""
        if self.debug:
            print('BUILDING OUTPUTS')
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
        self.out_offset = np.zeros(self.numOutputs)
        self.out_linear = np.zeros(self.numOutputs)
        self.out_quad = np.zeros(self.numOutputs)
        self.out_cubic = np.zeros(self.numOutputs)
        self.outputs_raw = np.zeros(self.numOutputs)
        for out in range(len(network.outputs)):  # iterate over the connections in the network
            sourcePop = network.outputs[out]['source']  # get the source
            self.out_offset[out] = network.outputs[out]['offset']
            self.out_linear[out] = network.outputs[out]['linear']
            self.out_quad[out] = network.outputs[out]['quadratic']
            self.out_cubic[out] = network.outputs[out]['cubic']
            for i in range(len(popsAndNrns[sourcePop])):
                self.outputVoltageConnectivity[outputs[out][i]][popsAndNrns[sourcePop][i]] = 1.0  # set the weight in the correct source and destination
                self.out_offset[outputs[out][i]] = network.outputs[out]['offset']
                self.out_linear[outputs[out][i]] = network.outputs[out]['linear']
                self.out_quad[outputs[out][i]] = network.outputs[out]['quadratic']
                self.out_cubic[outputs[out][i]] = network.outputs[out]['cubic']
        if self.debug:
            print('Input Connectivity:')
            print(self.inputConnectivity)
            print('GmaxNon:')
            print(self.GmaxNon)
            print('DelE:')
            print(self.DelE)
            print('Output Voltage Connectivity')
            print(self.outputVoltageConnectivity)
            print('U:')
            print(self.U)
            print('Ulast:')
            print(self.Ulast)
            print('\nDONE BUILDING')

    def forward(self, inputs) -> Any:
        self.Ulast = np.copy(self.U)
        self.inputs_mapped = self.in_cubic*(inputs**3) + self.in_quad*(inputs**2) + self.in_linear*inputs + self.in_offset
        Iapp = np.matmul(self.inputConnectivity, self.inputs_mapped)  # Apply external current sources to their destinations
        Gsyn = np.maximum(0, np.minimum(self.GmaxNon * self.Ulast / self.R, self.GmaxNon))
        Isyn = np.sum(Gsyn * self.DelE, axis=1) - self.Ulast * np.sum(Gsyn, axis=1)
        self.U = self.Ulast + self.timeFactorMembrane * (-self.Gm * self.Ulast + self.Ib + Isyn + Iapp)  # Update membrane potential
        self.outputs_raw = np.matmul(self.outputVoltageConnectivity, self.U)

        return self.out_cubic*(self.outputs_raw**3) + self.out_quad*(self.outputs_raw**2)\
            + self.out_linear*self.outputs_raw + self.out_offset

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
PYTORCH DENSE

Simulating the network using GPU-compatible tensors.
Note that this is not sparse, so memory may explode for large networks
"""

class SNS_Torch(Backend):
    # TODO: Add polynomial mapping
    def __init__(self, network: Network,device: str = 'cuda', **kwargs):
        super().__init__(network, **kwargs)

        """Neurons"""
        if self.debug:
            print('BUILDING NEURONS')
        # Initialize the vectors
        self.U = torch.from_numpy(np.zeros(self.numNeurons))
        self.Ulast = torch.from_numpy(np.zeros(self.numNeurons))
        self.spikes = torch.from_numpy(np.zeros(self.numNeurons))
        Cm = torch.from_numpy(np.zeros(self.numNeurons))
        self.Gm = torch.from_numpy(np.zeros(self.numNeurons))
        self.Ib = torch.from_numpy(np.zeros(self.numNeurons))
        self.theta0 = torch.from_numpy(np.zeros(self.numNeurons))
        self.theta = torch.from_numpy(np.zeros(self.numNeurons))
        self.thetaLast = torch.from_numpy(np.zeros(self.numNeurons))
        self.m = torch.from_numpy(np.zeros(self.numNeurons))
        tauTheta = torch.from_numpy(np.zeros(self.numNeurons))

        # iterate over the populations in the network
        popsAndNrns = []
        index = 0
        for pop in range(len(network.populations)):
            numNeurons = network.populations[pop]['number']  # find the number of neurons in the population
            popsAndNrns.append([])
            Ulast = 0.0
            for num in range(numNeurons):  # for each neuron, copy the parameters over
                Cm[index] = network.populations[pop]['type'].params['membraneCapacitance']
                self.Gm[index] = network.populations[pop]['type'].params['membraneConductance']
                self.Ib[index] = network.populations[pop]['type'].params['bias']
                self.Ulast[index] = Ulast
                if isinstance(network.populations[pop]['type'], SpikingNeuron):  # if the neuron is spiking, copy more
                    self.theta0[index] = network.populations[pop]['type'].params['thresholdInitialValue']
                    Ulast += network.populations[pop]['type'].params['thresholdInitialValue'] / numNeurons
                    self.m[index] = network.populations[pop]['type'].params['thresholdProportionalityConstant']
                    tauTheta[index] = network.populations[pop]['type'].params['thresholdTimeConstant']
                else:  # otherwise, set to the special values for NonSpiking
                    self.theta0[index] = sys.float_info.max
                    self.m[index] = 0
                    tauTheta[index] = 1
                    Ulast += self.R / numNeurons
                popsAndNrns[pop].append(index)
                index += 1
        self.U = self.Ulast.clone()
        # set the derived vectors
        self.timeFactorMembrane = self.dt / Cm
        self.timeFactorThreshold = self.dt / tauTheta
        self.theta = self.theta0.clone()
        self.thetaLast = self.theta0.clone()

        """Inputs"""
        if self.debug:
            print('BUILDING INPUTS')
        self.inputConnectivity = torch.from_numpy(np.zeros([self.numNeurons, self.numInputs]))  # initialize connectivity matrix
        for conn in network.inputConns:  # iterate over the connections in the network
            wt = conn['weight']  # get the weight
            source = conn['source']  # get the source
            destPop = conn['destination']  # get the destination
            for dest in popsAndNrns[destPop]:
                self.inputConnectivity[dest,source] = wt  # set the weight in the correct source and destination

        """Synapses"""
        if self.debug:
            print('BUILDING SYNAPSES')
        # initialize the matrices
        self.GmaxNon = torch.from_numpy(np.zeros([self.numNeurons, self.numNeurons]))
        # self.zeros = self.GmaxNon.clone()
        self.GmaxSpk = torch.from_numpy(np.zeros([self.numNeurons, self.numNeurons]))
        self.Gspike = torch.from_numpy(np.zeros([self.numNeurons, self.numNeurons]))
        self.DelE = torch.from_numpy(np.zeros([self.numNeurons, self.numNeurons]))
        self.tauSyn = torch.from_numpy(np.zeros([self.numNeurons, self.numNeurons]))+1

        # iterate over the synapses in the network
        for syn in range(len(network.synapses)):
            sourcePop = network.synapses[syn]['source']
            destPop = network.synapses[syn]['destination']
            Gmax = network.synapses[syn]['type'].params['maxConductance']
            delE = network.synapses[syn]['type'].params['relativeReversalPotential']

            if isinstance(network.synapses[syn]['type'], SpikingSynapse):
                tauS = network.synapses[syn]['type'].params['synapticTimeConstant']
                for source in popsAndNrns[sourcePop]:
                    for dest in popsAndNrns[destPop]:
                        self.GmaxSpk[dest,source] = Gmax / len(popsAndNrns[sourcePop])
                        self.DelE[dest,source] = delE
                        self.tauSyn[dest,source] = tauS
            else:
                for source in popsAndNrns[sourcePop]:
                    for dest in popsAndNrns[destPop]:
                        self.GmaxNon[dest,source] = Gmax / len(popsAndNrns[sourcePop])
                        self.DelE[dest,source] = delE
        self.timeFactorSynapse = self.dt / self.tauSyn

        """Outputs"""
        if self.debug:
            print('BUILDING OUTPUTS')
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

        self.outputVoltageConnectivity = torch.from_numpy(np.zeros([self.numOutputs, self.numNeurons]))  # initialize connectivity matrix
        self.outputSpikeConnectivity = self.outputVoltageConnectivity.clone()
        for out in range(len(network.outputs)):  # iterate over the connections in the network
            wt = network.outputs[out]['weight']  # get the weight
            sourcePop = network.outputs[out]['source']  # get the source
            for i in range(len(popsAndNrns[sourcePop])):
                if network.outputs[out]['spiking']:
                    self.outputSpikeConnectivity[outputs[out][i]][popsAndNrns[sourcePop][i]] = wt  # set the weight in the correct source and destination
                else:
                    self.outputVoltageConnectivity[outputs[out][i]][popsAndNrns[sourcePop][i]] = wt  # set the weight in the correct source and destination

        """Debug Prints"""
        if self.debug:
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
            print('U:')
            print(self.U)
            print('Ulast:')
            print(self.Ulast)
            print('theta0:')
            print(self.theta0)
            print('ThetaLast:')
            print(self.thetaLast)
            print('Theta')
            print(self.theta)
            print('\nDONE BUILDING')

        """Move the tensors to the appropriate device"""
        if device == 'cuda':
            if torch.cuda.is_available():
                if self.debug:
                    print("CUDA Device found, using GPU")
                self.device = 'cuda'
            else:
                if self.debug:
                    print('CUDA Device not found, using CPU')
                self.device = 'cpu'
        else:
            self.device = 'cpu'

        self.Ulast = self.Ulast.to(self.device)
        self.U = self.U.to(self.device)
        self.theta = self.theta.to(self.device)
        self.thetaLast = self.thetaLast.to(self.device)
        self.inputConnectivity = self.inputConnectivity.to(self.device)
        self.GmaxNon = self.GmaxNon.to(self.device)
        self.GmaxSpk = self.GmaxSpk.to(self.device)
        self.timeFactorSynapse = self.timeFactorSynapse.to(self.device)
        self.Gspike = self.Gspike.to(self.device)
        self.DelE = self.DelE.to(self.device)
        self.timeFactorMembrane = self.timeFactorMembrane.to(self.device)
        self.Gm = self.Gm.to(self.device)
        self.Ib = self.Ib.to(self.device)
        self.theta0 = self.theta0.to(self.device)
        self.timeFactorThreshold = self.timeFactorThreshold.to(self.device)
        self.m = self.m.to(self.device)
        self.spikes = self.spikes.to(self.device)
        self.zero = torch.tensor([0],device=self.device)
        self.outputSpikeConnectivity = self.outputSpikeConnectivity.to(self.device)
        self.outputVoltageConnectivity = self.outputVoltageConnectivity.to(self.device)
        self.out = torch.from_numpy(np.zeros([self.numOutputs, self.numNeurons])).to(self.device)
        self.Iapp = torch.from_numpy(np.zeros(self.numNeurons)).to(self.device)
        self.Gnon = torch.from_numpy(np.zeros([self.numNeurons, self.numNeurons])).to(self.device)
        self.Gsyn = torch.from_numpy(np.zeros([self.numNeurons, self.numNeurons])).to(self.device)
        self.Isyn = torch.from_numpy(np.zeros(self.numNeurons)).to(self.device)

    def forward(self, inputs) -> Any:
        self.Ulast = self.U.clone()
        self.thetaLast = self.theta.clone()
        # Iapp = torch.matmul(self.inputConnectivity, torch.from_numpy(inputs).to(self.device))  # Apply external current sources to their destinations
        self.Iapp = torch.matmul(self.inputConnectivity, inputs)  # Apply external current sources to their destinations
        self.Gnon = torch.maximum(self.zero, torch.minimum(self.GmaxNon * self.Ulast / self.R, self.GmaxNon))
        self.Gspike = self.Gspike * (1 - self.timeFactorSynapse)
        self.Gsyn = self.Gnon + self.Gspike
        self.Isyn = torch.sum(self.Gsyn * self.DelE, dim=1) - self.Ulast * torch.sum(self.Gsyn, dim=1)
        self.U = self.Ulast + self.timeFactorMembrane * (
                    -self.Gm * self.Ulast + self.Ib + self.Isyn + self.Iapp)  # Update membrane potential
        self.theta = self.thetaLast + self.timeFactorThreshold * (
                    -self.thetaLast + self.theta0 + self.m * self.Ulast)  # Update the firing thresholds
        self.spikes = torch.sign(torch.minimum(self.zero, self.theta - self.U))  # Compute which neurons have spiked
        self.Gspike = torch.maximum(self.Gspike,
                                 (-self.spikes) * self.GmaxSpk)  # Update the conductance of synapses which spiked
        self.U = self.U * (self.spikes + 1)  # Reset the membrane voltages of neurons which spiked
        self.out = torch.matmul(self.outputVoltageConnectivity, self.U) + torch.matmul(self.outputSpikeConnectivity, -self.spikes)
        # return out.cpu().numpy()
        return self.out

"""
########################################################################################################################
PYTORCH SPARSE
"""

class SNS_Torch_Large(Backend):
    # TODO: Add polynomial mapping
    def __init__(self, network: Network,dtype, **kwargs):
        super().__init__(network, **kwargs)

        #Neurons
        if self.debug:
            print('BUILDING NEURONS')
        # Initialize the vectors
        self.U = torch.from_numpy(np.zeros(self.numNeurons)).to(dtype)
        self.Ulast = torch.from_numpy(np.zeros(self.numNeurons)).to(dtype)
        self.spikes = torch.from_numpy(np.zeros(self.numNeurons)).to(dtype)
        Cm = torch.from_numpy(np.zeros(self.numNeurons)).to(dtype)
        self.Gm = torch.from_numpy(np.zeros(self.numNeurons)).to(dtype)
        self.Ib = torch.from_numpy(np.zeros(self.numNeurons)).to(dtype)
        self.theta0 = torch.from_numpy(np.zeros(self.numNeurons)).to(dtype)
        self.theta = torch.from_numpy(np.zeros(self.numNeurons)).to(dtype)
        self.thetaLast = torch.from_numpy(np.zeros(self.numNeurons)).to(dtype)
        self.m = torch.from_numpy(np.zeros(self.numNeurons)).to(dtype)
        tauTheta = torch.from_numpy(np.zeros(self.numNeurons)).to(dtype)

        # iterate over the populations in the network
        popsAndNrns = []
        index = 0
        for pop in range(len(network.populations)):
            numNeurons = network.populations[pop]['number']  # find the number of neurons in the population
            popsAndNrns.append([])
            Ulast = 0.0
            for num in range(numNeurons):  # for each neuron, copy the parameters over
                Cm[index] = network.populations[pop]['type'].params['membraneCapacitance']
                self.Gm[index] = network.populations[pop]['type'].params['membraneConductance']
                self.Ib[index] = network.populations[pop]['type'].params['bias']
                self.Ulast[index] = Ulast
                if isinstance(network.populations[pop]['type'], SpikingNeuron):  # if the neuron is spiking, copy more
                    self.theta0[index] = network.populations[pop]['type'].params['thresholdInitialValue']
                    Ulast += network.populations[pop]['type'].params['thresholdInitialValue'] / numNeurons
                    self.m[index] = network.populations[pop]['type'].params['thresholdProportionalityConstant']
                    tauTheta[index] = network.populations[pop]['type'].params['thresholdTimeConstant']
                else:  # otherwise, set to the special values for NonSpiking
                    self.theta0[index] = sys.float_info.max
                    self.m[index] = 0
                    tauTheta[index] = 1
                    Ulast += self.R / numNeurons
                popsAndNrns[pop].append(index)
                index += 1
        self.U = self.Ulast.clone().to(dtype)
        # set the derived vectors
        self.timeFactorMembrane = (self.dt / Cm).to(dtype)
        self.timeFactorThreshold = (self.dt / tauTheta).to(dtype)
        self.theta = self.theta0.clone().to(dtype)
        self.thetaLast = self.theta0.clone().to(dtype)

        #Inputs
        if self.debug:
            print('BUILDING INPUTS')
        # self.inputConnectivity = torch.from_numpy(np.zeros([self.numNeurons, self.numInputs]))  # initialize connectivity matrix
        rows = []
        cols = []
        vals = []
        for conn in network.inputConns:  # iterate over the connections in the network
            wt = conn['weight']  # get the weight
            source = conn['source']  # get the source
            destPop = conn['destination']  # get the destination
            for dest in popsAndNrns[destPop]:
                rows.append(dest)
                cols.append(source)
                vals.append(wt)
                # self.inputConnectivity[dest,source] = wt  # set the weight in the correct source and destination
        self.inputConnectivity = torch.sparse_coo_tensor([rows,cols],vals,(self.numNeurons,self.numInputs)).to(dtype)

        #Synapses
        if self.debug:
            print('BUILDING SYNAPSES')
        # initialize the matrices
        # self.GmaxNon = torch.from_numpy(np.zeros([self.numNeurons, self.numNeurons]))
        # self.zeros = torch.sparse_coo_tensor(size=(self.numNeurons,self.numNeurons))
        # self.GmaxSpk = torch.from_numpy(np.zeros([self.numNeurons, self.numNeurons]))

        # self.DelE = torch.from_numpy(np.zeros([self.numNeurons, self.numNeurons]))
        self.tauSyn = (torch.from_numpy(np.zeros([self.numNeurons, self.numNeurons]))+1).to(dtype)

        # iterate over the synapses in the network
        nonRows = []
        nonCols = []
        nonVals = []
        spikeRows = []
        spikeCols = []
        spikeVals = []
        spikeCond = []
        delERows = []
        delECols = []
        delEVals = []
        ones = []
        for syn in range(len(network.synapses)):
            sourcePop = network.synapses[syn]['source']
            destPop = network.synapses[syn]['destination']
            Gmax = network.synapses[syn]['type'].params['maxConductance']
            delE = network.synapses[syn]['type'].params['relativeReversalPotential']

            if isinstance(network.synapses[syn]['type'], SpikingSynapse):
                tauS = network.synapses[syn]['type'].params['synapticTimeConstant']
                for source in popsAndNrns[sourcePop]:
                    for dest in popsAndNrns[destPop]:
                        # self.GmaxSpk[dest,source] = Gmax / len(popsAndNrns[sourcePop])
                        # self.DelE[dest,source] = delE
                        self.tauSyn[dest,source] = tauS
                        spikeRows.append(dest)
                        spikeCols.append(source)
                        spikeVals.append(Gmax / len(popsAndNrns[sourcePop]))
                        delERows.append(dest)
                        delECols.append(source)
                        delEVals.append(delE)
                        spikeCond.append(0.0)
                        ones.append(1.0)
            else:
                for source in popsAndNrns[sourcePop]:
                    for dest in popsAndNrns[destPop]:
                        # self.GmaxNon[dest,source] = Gmax / len(popsAndNrns[sourcePop])
                        # self.DelE[dest,source] = delE
                        nonRows.append(dest)
                        nonCols.append(source)
                        nonVals.append(Gmax / len(popsAndNrns[sourcePop]))
                        delERows.append(dest)
                        delECols.append(source)
                        delEVals.append(delE)
        self.GmaxNon = torch.sparse_coo_tensor([nonRows,nonCols],nonVals,(self.numNeurons,self.numNeurons)).to(dtype)
        self.GmaxSpk = torch.sparse_coo_tensor([spikeRows, spikeCols], spikeVals, (self.numNeurons, self.numNeurons)).to(dtype)
        self.DelE = torch.sparse_coo_tensor([delERows, delECols], delEVals, (self.numNeurons, self.numNeurons)).to(dtype)
        self.Gspike = torch.sparse_coo_tensor([spikeRows,spikeCols],spikeCond,size=(self.numNeurons, self.numNeurons)).to(dtype)
        self.ones = torch.sparse_coo_tensor([spikeRows,spikeCols],ones,size=(self.numNeurons,self.numNeurons)).to(dtype)
        self.timeFactorSynapse = (self.dt / self.tauSyn).to(dtype)

        #Outputs
        if self.debug:
            print('BUILDING OUTPUTS')
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

        # self.outputVoltageConnectivity = torch.from_numpy(np.zeros([self.numOutputs, self.numNeurons]))  # initialize connectivity matrix
        # self.outputSpikeConnectivity = self.outputVoltageConnectivity.clone()
        voltRows = []
        voltCols = []
        voltVals = []
        spikeRows = []
        spikeCols = []
        spikeVals = []
        for out in range(len(network.outputs)):  # iterate over the connections in the network
            wt = network.outputs[out]['weight']  # get the weight
            sourcePop = network.outputs[out]['source']  # get the source
            for i in range(len(popsAndNrns[sourcePop])):
                if network.outputs[out]['spiking']:
                    # self.outputSpikeConnectivity[outputs[out][i]][popsAndNrns[sourcePop][i]] = wt  # set the weight in the correct source and destination
                    spikeRows.append(outputs[out][i])
                    spikeCols.append(popsAndNrns[sourcePop][i])
                    spikeVals.append(wt)
                else:
                    # self.outputVoltageConnectivity[outputs[out][i]][popsAndNrns[sourcePop][i]] = wt  # set the weight in the correct source and destination
                    voltRows.append(outputs[out][i])
                    voltCols.append(popsAndNrns[sourcePop][i])
                    voltVals.append(wt)
        self.outputVoltageConnectivity = torch.sparse_coo_tensor([voltRows,voltCols],voltVals,(self.numOutputs,self.numNeurons)).to(dtype)
        self.outputSpikeConnectivity = torch.sparse_coo_tensor([spikeRows,spikeCols],spikeVals,(self.numOutputs,self.numNeurons)).to(dtype)

        #DEBUG
        if self.debug:
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
            print('U:')
            print(self.U)
            print('Ulast:')
            print(self.Ulast)
            print('theta0:')
            print(self.theta0)
            print('ThetaLast:')
            print(self.thetaLast)
            print('Theta')
            print(self.theta)
            print('\nDONE BUILDING')

        #Move the tensors
        if torch.cuda.is_available():
            if self.debug:
                print("CUDA Device found, using GPU")
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # self.Ulast = self.Ulast.to(self.device)
        # self.U = self.U.to(self.device)
        # self.theta = self.theta.to(self.device)
        # self.thetaLast = self.thetaLast.to(self.device)
        # self.inputConnectivity = self.inputConnectivity.to(self.device).double()
        self.inputConnectivity = self.inputConnectivity.to(dtype)
        # self.GmaxNon = self.GmaxNon.to(self.device)
        # self.GmaxSpk = self.GmaxSpk.to(self.device)
        # self.timeFactorSynapse = self.timeFactorSynapse.to(self.device)
        # self.Gspike = self.Gspike.to(self.device)
        # self.DelE = self.DelE.to(self.device)
        # self.timeFactorMembrane = self.timeFactorMembrane.to(self.device)
        # self.Gm = self.Gm.to(self.device)
        # self.Ib = self.Ib.to(self.device)
        # self.theta0 = self.theta0.to(self.device)
        # self.timeFactorThreshold = self.timeFactorThreshold.to(self.device)
        # self.m = self.m.to(self.device)
        # self.spikes = self.spikes.to(self.device)
        # # self.zeros = self.zeros.to(self.device)
        # self.ones = self.ones.to(self.device)
        # # self.zeros1d = torch.from_numpy(np.zeros(self.numNeurons)).to(self.device)
        # self.outputSpikeConnectivity = self.outputSpikeConnectivity.to(self.device).double()
        # self.outputVoltageConnectivity = self.outputVoltageConnectivity.to(self.device).double()
        self.outputSpikeConnectivity = self.outputSpikeConnectivity.to(dtype)
        self.outputVoltageConnectivity = self.outputVoltageConnectivity.to(dtype)
        self.zero = torch.tensor([0.0],device='cpu',dtype=dtype)

    def forward(self, inputs) -> Any:
        self.Ulast = self.U.clone()
        self.thetaLast = self.theta.clone()

        # Iapp = torch.matmul(self.inputConnectivity, torch.from_numpy(inputs).to(self.device))  # Apply external current sources to their destinations
        [self.inputConnectivity,inputs] = sendVars([self.inputConnectivity,inputs],self.device)
        Iapp = torch.matmul(self.inputConnectivity, inputs)  # Apply external current sources to their destinations
        [self.inputConnectivity,inputs,Iapp] = sendVars([self.inputConnectivity, inputs,Iapp], 'cpu')

        [self.zero,self.GmaxNon,self.Ulast] = sendVars([self.zero,self.GmaxNon,self.Ulast],self.device)
        Gnon = torch.maximum(self.zero, torch.minimum(self.GmaxNon.to_dense() * (self.Ulast / self.R), self.GmaxNon.to_dense())).to_sparse()    # Sparse version unsupported
        [self.zero, self.GmaxNon, self.Ulast, Gnon] = sendVars([self.zero, self.GmaxNon, self.Ulast, Gnon], 'cpu')

        [self.Gspike,self.ones,self.timeFactorSynapse] = sendVars([self.Gspike,self.ones,self.timeFactorSynapse],self.device)
        self.Gspike = self.Gspike * (self.ones - self.timeFactorSynapse.to_sparse())    # Sparse version unsupported
        [self.ones, self.timeFactorSynapse] = sendVars([self.ones, self.timeFactorSynapse], 'cpu')

        [Gnon] = sendVars([Gnon],self.device)
        Gsyn = Gnon + self.Gspike
        [Gnon,self.Gspike] = sendVars([Gnon,self.Gspike],'cpu')

        [self.DelE,self.Ulast] = sendVars([self.DelE,self.Ulast],self.device)
        Isyn = (torch.sum(Gsyn.to_dense() * self.DelE.to_dense(), dim=1)).to_sparse() - (self.Ulast * torch.sum(Gsyn.to_dense(), dim=1)).to_sparse()    # Sparse version unsupported
        [Gsyn,self.DelE] = sendVars([Gsyn,self.DelE],'cpu')

        [self.U,self.timeFactorMembrane,self.Gm,self.Ib,Iapp] = sendVars([self.U,self.timeFactorMembrane,self.Gm,self.Ib,Iapp],self.device)
        self.U = self.Ulast + self.timeFactorMembrane * (
                    -self.Gm * self.Ulast + self.Ib + Isyn + Iapp)  # Update membrane potential
        [self.U, self.timeFactorMembrane, self.Gm, self.Ib, Iapp,Isyn] = sendVars(
            [self.U, self.timeFactorMembrane, self.Gm, self.Ib, Iapp,Isyn], 'cpu')

        [self.theta,self.thetaLast,self.timeFactorThreshold,self.theta0,self.m] = sendVars([self.theta,self.thetaLast,self.timeFactorThreshold,self.theta0,self.m],self.device)
        self.theta = self.thetaLast + self.timeFactorThreshold * (
                    -self.thetaLast + self.theta0 + self.m * self.Ulast)  # Update the firing thresholds
        [self.thetaLast, self.timeFactorThreshold, self.theta0, self.m, self.Ulast] = sendVars(
            [self.thetaLast, self.timeFactorThreshold, self.theta0, self.m,self.Ulast], 'cpu')

        [self.spikes,self.zero,self.U] = sendVars([self.spikes,self.zero,self.U],self.device)
        self.spikes = torch.sign(torch.minimum(self.zero, self.theta - self.U))  # Compute which neurons have spiked
        [self.zero,self.theta,self.U] = sendVars([self.zero,self.theta,self.U],'cpu')

        [self.Gspike,self.GmaxSpk] = sendVars([self.Gspike,self.GmaxSpk],self.device)
        self.Gspike = torch.maximum(self.Gspike.to_dense(),(-self.spikes) * self.GmaxSpk.to_dense()).to_sparse()  # Update the conductance of synapses which spiked, sparse version unsupported
        [self.Gspike,self.GmaxSpk] = sendVars([self.Gspike,self.GmaxSpk],'cpu')

        [self.U] = sendVars([self.U],self.device)
        self.U = self.U * (self.spikes + 1)  # Reset the membrane voltages of neurons which spiked

        [self.outputVoltageConnectivity,self.outputSpikeConnectivity] = sendVars([self.outputVoltageConnectivity,self.outputSpikeConnectivity],self.device)
        out = torch.matmul(self.outputVoltageConnectivity, self.U) + torch.matmul(self.outputSpikeConnectivity, -self.spikes)
        [out,self.outputVoltageConnectivity,self.U, self.outputSpikeConnectivity,self.spikes] = sendVars(
            [out,self.outputVoltageConnectivity,self.U, self.outputSpikeConnectivity,self.spikes],'cpu')
        # return out.cpu().numpy()
        return out
