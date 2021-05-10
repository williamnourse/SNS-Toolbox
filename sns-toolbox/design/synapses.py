"""
The mechanism for defining a synapse (connection) model which can be simulated in the SNS Toolbox
William Nourse
May 10, 2021
He's convinced me, gimme back my dollar!
"""

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
IMPORTS
"""

from typing import Dict, Any, Optional

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
BASE CLASS
"""

class Synapse:
    def __init__(self, name: str = 'Synapse') -> None:
        """
        Constructor for the base class of all synapses
        :param name: Name of this synapse preset
        """
        self.params: Dict[str, Any] = {}
        # TODO: Type checking
        self.params['name'] = name

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SPECIFIC MODELS
"""

class NonSpikingSynapse(Synapse):
    def __init__(self,maxConductance: float = 1.0,
                 relativeReversalPotential: float = 40.0,
                 R: float = 20.0,
                 **kwargs: Any) -> None:
        """
        Basic non-spiking synapse, where the conductance is defined as the following:
        Conductance = maxConductance * max(0, min(1, Upre/R)), and the synaptic current is
        Isyn = Conductance*(relativeReversalPotential - Upost)
        :param maxConductance:              uS
        :param relativeReversalPotential:   mV
        :param R:                           mV
        """
        super().__init__(**kwargs)  # Call to constructor of parent class
        # TODO: Type checking
        self.params['maxConductance'] = maxConductance
        self.params['relativeReversalPotential'] = relativeReversalPotential
        self.params['R'] = R

class TransmissionSynapse(NonSpikingSynapse):
    def __init__(self, gain: float = 1.0,
                 name: str = 'Transmit',
                 **kwargs) -> None:
        """
        Transmission synapse, where (given some gain) the maximum conductunce is
        maxConductance = (gain*R)/(relativeReversalPotential - gain*R)
        :param gain:    Transmission gain
        :param name:    Name of this synapse preset
        """
        super().__init__(name=name, **kwargs)  # Call to constructor of parent class
        self.params['maxConductance'] = (gain*self.params['R'])/\
                                        (self.params['relativeReversalPotential']-gain*self.params['R'])

class ModulationSynapse(NonSpikingSynapse):
    def __init__(self, name: str = 'Modulate', **kwargs) -> None:
        """
        Modulation synapse, where the relativeReversalPotential is set to 0
        :param name: Name of this synapse preset
        """
        super().__init__(name=name,**kwargs)
        self.params['relativeReversalPotential'] = 0.0

