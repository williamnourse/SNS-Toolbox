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
    def __init__(self, max_conductance: float = 1.0,
                 relative_reversal_potential: float = 40.0,
                 **kwargs: Any) -> None:
        """
        Basic non-spiking synapse, where the conductance is defined as the following:
        Conductance = max_conductance * max(0, min(1, Upre/R)), and the synaptic current is
        i_syn = Conductance*(relative_reversal_potential - Upost)
        :param max_conductance:              uS
        :param relative_reversal_potential:   mV
        """
        super().__init__(**kwargs)  # Call to constructor of parent class
        if isinstance(max_conductance, numbers.Number):
            if max_conductance > 0:
                self.params['max_conductance'] = max_conductance
            else:
                raise ValueError('max_conductance (gMax) must be greater than 0')
        else:
            raise TypeError('max_conductance (gMax) must be a number (int, float, double, etc.')
        if isinstance(relative_reversal_potential, numbers.Number):
            self.params['relative_reversal_potential'] = relative_reversal_potential
        else:
            raise TypeError('relative_reversal_potential (deltaEsyn) must be a number (int, float, double, etc.')

class SpikingSynapse(Synapse):
    def __init__(self, max_conductance: float = 1.0,
                 relative_reversal_potential: float = 194.0,
                 time_constant: float = 1.0,
                 transmission_delay: int = 0,
                 R: float = 20.0,
                 **kwargs: Any) -> None:
        """
        Basic non-spiking synapse, where the conductance is defined as the following:
        Conductance = max_conductance * max(0, min(1, Upre/R)), and the synaptic current is
        i_syn = Conductance*(relative_reversal_potential - Upost)
        :param max_conductance:              uS
        :param relative_reversal_potential:   mV
        :param R:                           mV
        """
        super().__init__(**kwargs)  # Call to constructor of parent class
        if isinstance(max_conductance, numbers.Number):
            if max_conductance > 0:
                self.params['max_conductance'] = max_conductance
            else:
                raise ValueError('max_conductance (gMax) must be greater than 0')
        else:
            raise TypeError('max_conductance (gMax) must be a number (int, float, double, etc.')
        if isinstance(relative_reversal_potential, numbers.Number):
            self.params['relative_reversal_potential'] = relative_reversal_potential
        else:
            raise TypeError('relative_reversal_potential (deltaEsyn) must be a number (int, float, double, etc.')
        if isinstance(time_constant, numbers.Number):
            if time_constant > 0:
                self.params['synapticTimeConstant'] = time_constant
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
        if isinstance(transmission_delay, int):
            if transmission_delay >= 0:
                self.params['synapticTransmissionDelay'] = transmission_delay
            else:
                raise ValueError('Synaptic transmission delay must be greater than or equal to zero')
        else:
            raise TypeError('Synaptic transmission delay must be an integer')

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
        max_conductance = (gain*R)/(relative_reversal_potential - gain*R)
        :param gain:    Transmission gain
        :param name:    Name of this synapse preset
        """
        super().__init__(name=name, **kwargs)  # Call to constructor of parent class
        if isinstance(gain,numbers.Number):
            if gain == 0:
                raise ValueError('Gain must be nonzero')
            elif math.copysign(1,gain) != math.copysign(1,self.params['relative_reversal_potential']):    # sign of gain and DeltaE don't match
                raise ValueError('Gain of '+str(gain)+' and Relative Reversal Potential must have the same sign')
            else:
                try:
                    self.params['max_conductance'] = (gain*R)/(self.params['relative_reversal_potential']-gain*R)
                except ZeroDivisionError:
                    raise ValueError('Gain of '+str(gain)+' causes division by 0, decrease gain or increase relative_reversal_potential')
                if self.params['max_conductance'] < 0:
                    raise ValueError('Gain of '+str(gain)+' causes max_conductance to be negative, decrease gain or increase relative_reversal_potential')
        else:
            raise TypeError('Gain of '+str(gain)+' must be a number (int, float, double, etc.)')

class NonSpikingModulationSynapse(NonSpikingSynapse):
    def __init__(self,ratio, name: str = 'Modulate', **kwargs) -> None:
        """
        Modulation synapse, where the relative_reversal_potential is set to 0
        :param name: Name of this synapse preset
        """
        super().__init__(name=name,**kwargs)
        self.params['relative_reversal_potential'] = 0.0
        self.params['max_conductance'] = 1/ratio - 1

