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

from typing import Dict, Any
import numbers

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
        if isinstance(name,str):
            self.params['name'] = name
        else:
            raise ValueError('Name should be a string')

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
        if isinstance(maxConductance,numbers.Number):
            if maxConductance > 0:
                self.params['maxConductance'] = maxConductance
            else:
                raise ValueError('maxConductance (gMax) must be greater than 0')
        else:
            raise ValueError('maxConductance (gMax) must be a number (int, float, double, etc.')
        if isinstance(relativeReversalPotential,numbers.Number):
            self.params['relativeReversalPotential'] = relativeReversalPotential
        else:
            raise ValueError('relativeReversalPotential (deltaEsyn) must be a number (int, float, double, etc.')
        if isinstance(R,numbers.Number):
            if R > 0:
                self.params['R'] = R
            else:
                raise ValueError('R must be greater than 0')
        else:
            raise ValueError('R must be a number (int, float, double, etc.')

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
        if isinstance(gain,numbers.Number):
            if gain <= 0:
                raise ValueError('Gain must be greater than 0')
            else:
                try:
                    self.params['maxConductance'] = (gain*self.params['R'])/(self.params['relativeReversalPotential']-gain*self.params['R'])
                except ZeroDivisionError:
                    raise ValueError('Gain causes division by 0, decrease gain or increase relativeReversalPotential')
                if self.params['maxConductance'] < 0:
                    raise ValueError('Gain causes maxConductance to be negative, decrease gain or increase relativeReversalPotential')
        else:
            raise ValueError('Gain must be a number (int, float, double, etc.)')

class ModulationSynapse(NonSpikingSynapse):
    def __init__(self, name: str = 'Modulate', **kwargs) -> None:
        """
        Modulation synapse, where the relativeReversalPotential is set to 0
        :param name: Name of this synapse preset
        """
        super().__init__(name=name,**kwargs)
        self.params['relativeReversalPotential'] = 0.0

