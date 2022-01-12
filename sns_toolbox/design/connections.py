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

class PatternConnection:
    def __init__(self,gain_matrix,
                 name: str = 'Pattern',
                 R: float = 20.0,
                 positive_reversal_potential: float = 160.0,
                 negative_reversal_potential: float = -80.0,
                 wrap: bool = False):
        """
        Connection pattern between two neural populations (i.e. a kernel)
        :param gain_matrix: Matrix (or vector for 1D kernels) of synaptic gains
        :param name:    Name of this connection type
        :param R:   Voltage range of neural activity (mV)
        :param positive_reversal_potential: Reversal potential for synapses with positive gain (mV)
        :param negative_reversal_potential: Reversal potential for synapses with negative gain (mV)
        :param wrap:    Flag for if connections should wrap from one end of the population to the other
        """

        self.params = {'R': R,
                       'name': name,
                       'wrap': wrap,
                       'maxConductance': [],
                       'relativeReversalPotential': []}

        self.positive_reversal_potential = positive_reversal_potential
        self.negative_reversal_potential = negative_reversal_potential

        if hasattr(gain_matrix[0],'__iter__'): # 2D kernel
            for row in range(len(gain_matrix)):
                cond_values = []
                del_e_values = []
                for col in range(len(gain_matrix[0])):
                    max_conductance, relative_reversal_potential = self.__calc_synaptic_parameters__(gain_matrix[row][col])
                    cond_values.append(max_conductance)
                    del_e_values.append(relative_reversal_potential)
                self.params['maxConductance'].append(cond_values)
                self.params['relativeReversalPotential'].append(del_e_values)
        else:   # 1D kernel
            for i in range(len(gain_matrix)):
                max_conductance, relative_reversal_potential = self.__calc_synaptic_parameters__(gain_matrix[i])
                self.params['maxConductance'].append(max_conductance)
                self.params['relativeReversalPotential'].append(relative_reversal_potential)

    def __calc_synaptic_parameters__(self,gain):
        if gain == 0.0:
            return 0.0, 0.0
        else:
            if gain > 0.0:
                relative_reversal_potential = self.positive_reversal_potential
            else:
                relative_reversal_potential = self.negative_reversal_potential
            max_conductance = gain*self.params['R']/(relative_reversal_potential-gain*self.params['R'])

            return max_conductance, relative_reversal_potential



"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SPECIFIC MODELS
"""

class NonSpikingTransmissionSynapse(NonSpikingSynapse):
    def __init__(self, gain: float = 1.0,
                 name: str = 'Transmit',
                 R: float = 20.0,
                 **kwargs) -> None:
        """
        Transmission synapse, where (given some integration_gain) the maximum conductance is
        max_conductance = (integration_gain*R)/(relative_reversal_potential - integration_gain*R)
        :param gain:    Transmission integration_gain
        :param name:    Name of this synapse preset
        """
        super().__init__(name=name, **kwargs)  # Call to constructor of parent class
        if isinstance(gain,numbers.Number):
            if gain == 0:
                raise ValueError('Gain must be nonzero')
            elif math.copysign(1,gain) != math.copysign(1,self.params['relative_reversal_potential']):    # sign of integration_gain and DeltaE don't match
                raise ValueError('Gain of '+str(gain)+' and Relative Reversal Potential must have the same sign')
            else:
                try:
                    self.params['max_conductance'] = (gain*R)/(self.params['relative_reversal_potential']-gain*R)
                except ZeroDivisionError:
                    raise ValueError('Gain of '+str(gain)+' causes division by 0, decrease integration_gain or increase relative_reversal_potential')
                if self.params['max_conductance'] < 0:
                    raise ValueError('Gain of '+str(gain)+' causes max_conductance to be negative, decrease integration_gain or increase relative_reversal_potential')
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

