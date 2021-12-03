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
import math

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
            self.name = name
        else:
            raise TypeError('Name should be a string')

class NonSpikingSynapse(Synapse):
    def __init__(self,maxConductance: float = 1.0,
                 relativeReversalPotential: float = 40.0,
                 **kwargs: Any) -> None:
        """
        Basic non-spiking synapse, where the conductance is defined as the following:
        Conductance = maxConductance * max(0, min(1, Upre/R)), and the synaptic current is
        Isyn = Conductance*(relativeReversalPotential - Upost)
        :param maxConductance:              uS
        :param relativeReversalPotential:   mV
        """
        super().__init__(**kwargs)  # Call to constructor of parent class
        if isinstance(maxConductance,numbers.Number):
            if maxConductance > 0:
                self.params['maxConductance'] = maxConductance
            else:
                raise ValueError('maxConductance (gMax) must be greater than 0')
        else:
            raise TypeError('maxConductance (gMax) must be a number (int, float, double, etc.')
        if isinstance(relativeReversalPotential,numbers.Number):
            self.params['relativeReversalPotential'] = relativeReversalPotential
        else:
            raise TypeError('relativeReversalPotential (deltaEsyn) must be a number (int, float, double, etc.')

class SpikingSynapse(Synapse):
    def __init__(self,maxConductance: float = 1.0,
                 relativeReversalPotential: float = 40.0,
                 synapticTimeConstant: float = 1.0,
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
            raise TypeError('maxConductance (gMax) must be a number (int, float, double, etc.')
        if isinstance(relativeReversalPotential,numbers.Number):
            self.params['relativeReversalPotential'] = relativeReversalPotential
        else:
            raise TypeError('relativeReversalPotential (deltaEsyn) must be a number (int, float, double, etc.')
        if isinstance(synapticTimeConstant,numbers.Number):
            if synapticTimeConstant > 0:
                self.params['synapticTimeConstant'] = synapticTimeConstant
            else:
                raise ValueError('Synaptic time constant (tauS) must be greater than 0')
        else:
            raise TypeError('Synaptic time constant (tauS) must be a number (int, float, double, etc.) greater than 0')
        if isinstance(R,numbers.Number):
            if R > 0:
                self.params['R'] = R
            else:
                raise ValueError('R must be greater than 0')
        else:
            raise TypeError('R must be a number (int, float, double, etc.')

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SPECIFIC MODELS
"""

class NonSpikingTransmissionSynapse(NonSpikingSynapse):
    def __init__(self, gain: float = 1.0,
                 name: str = 'Transmit',
                 R=20.0,
                 **kwargs) -> None:
        """
        Transmission synapse, where (given some gain) the maximum conductunce is
        maxConductance = (gain*R)/(relativeReversalPotential - gain*R)
        :param gain:    Transmission gain
        :param name:    Name of this synapse preset
        """
        super().__init__(name=name, **kwargs)  # Call to constructor of parent class
        if isinstance(gain,numbers.Number):
            if gain == 0:
                raise ValueError('Gain must be nonzero')
            elif math.copysign(1,gain) != math.copysign(1,self.params['relativeReversalPotential']):    # sign of gain and DeltaE don't match
                raise ValueError('Gain of '+str(gain)+' and Relative Reversal Potential must have the same sign')
            else:
                try:
                    self.params['maxConductance'] = (gain*R)/(self.params['relativeReversalPotential']-gain*R)
                except ZeroDivisionError:
                    raise ValueError('Gain of '+str(gain)+' causes division by 0, decrease gain or increase relativeReversalPotential')
                if self.params['maxConductance'] < 0:
                    raise ValueError('Gain of '+str(gain)+' causes maxConductance to be negative, decrease gain or increase relativeReversalPotential')
        else:
            raise TypeError('Gain of '+str(gain)+' must be a number (int, float, double, etc.)')

class NonSpikingModulationSynapse(NonSpikingSynapse):
    def __init__(self,ratio, name: str = 'Modulate', **kwargs) -> None:
        """
        Modulation synapse, where the relativeReversalPotential is set to 0
        :param name: Name of this synapse preset
        """
        super().__init__(name=name,**kwargs)
        self.params['relativeReversalPotential'] = 0.0
        self.params['maxConductance'] = 1/ratio - 1

