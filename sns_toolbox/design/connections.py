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
class Connection:
    def __init__(self, max_conductance, relative_reversal_potential, name: str = 'Connection'):
        """
        Constructor for the base class of all connections
        :param name:    Name of this connection type
        """
        self.params: Dict[str, Any] = {}
        self.params['spiking'] = False
        self.params['pattern'] = False
        self.params['max_conductance'] = max_conductance
        self.params['relative_reversal_potential'] = relative_reversal_potential
        if isinstance(name, str):
            self.params['name'] = name
        else:
            raise TypeError('Name should be a string')

class NonSpikingConnection(Connection):
    def __init__(self, max_conductance, relative_reversal_potential, name: str = 'Non-Spiking Connection') -> None:
        """
        Constructor for the base class of all non-spiking connections
        :param name: Name of this connection preset
        """
        super().__init__(max_conductance, relative_reversal_potential, name)
        self.params['pattern'] = False
        self.params['spiking'] = False

class SpikingConnection(Connection):
    def __init__(self, max_conductance, relative_reversal_potential, time_constant, transmission_delay, R,
                 name: str = 'Spiking Connection') -> None:
        """
        Constructor for the base class of all spiking connections
        :param name: Name of this connection preset
        """
        super().__init__(max_conductance, relative_reversal_potential, name)
        self.params['pattern'] = False
        self.params['spiking'] = True
        self.params['synapticTimeConstant'] = time_constant

class NonSpikingSynapse(NonSpikingConnection):
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
        if isinstance(max_conductance, numbers.Number):
            if max_conductance <= 0:
                raise ValueError('max_conductance (gMax) must be greater than 0')
        else:
            raise TypeError('max_conductance (gMax) must be a number (int, float, double, etc.')
        if not isinstance(relative_reversal_potential, numbers.Number):
            raise TypeError('relative_reversal_potential (deltaEsyn) must be a number (int, float, double, etc.')
        super().__init__(max_conductance, relative_reversal_potential,**kwargs)  # Call to constructor of parent class

class SpikingSynapse(SpikingConnection):
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
        if isinstance(max_conductance, numbers.Number):
            if max_conductance <= 0:
                raise ValueError('max_conductance (gMax) must be greater than 0')
        else:
            raise TypeError('max_conductance (gMax) must be a number (int, float, double, etc.')
        if not isinstance(relative_reversal_potential, numbers.Number):
            raise TypeError('relative_reversal_potential (deltaEsyn) must be a number (int, float, double, etc.')
        super().__init__(max_conductance, relative_reversal_potential, **kwargs)  # Call to constructor of parent class

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

class NonSpikingPatternConnection(NonSpikingConnection):
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
        :param positive_reversal_potential: Reversal potential for connections with positive gain (mV)
        :param negative_reversal_potential: Reversal potential for connections with negative gain (mV)
        :param wrap:    Flag for if connections should wrap from one end of the population to the other
        """
        max_conductance = []
        relative_reversal_potential = []
        super().__init__(max_conductance,relative_reversal_potential,name=name)
        self.params['wrap'] = wrap
        self.params['R'] = R
        self.params['pattern'] = True

        self.positive_reversal_potential = positive_reversal_potential
        self.negative_reversal_potential = negative_reversal_potential

        if hasattr(gain_matrix[0],'__iter__'): # 2D kernel
            for row in range(len(gain_matrix)):
                cond_values = []
                del_e_values = []
                for col in range(len(gain_matrix[0])):
                    calc_max_conductance,calc_relative_reversal_potential = __calc_synaptic_parameters_from_gain__(gain_matrix[row][col],
                                                                                                                   positive_reversal_potential,
                                                                                                                   negative_reversal_potential,
                                                                                                                   self.params['R'])
                    cond_values.append(calc_max_conductance)
                    del_e_values.append(calc_relative_reversal_potential)
                self.params['max_conductance'].append(cond_values)
                self.params['relative_reversal_potential'].append(del_e_values)
        else:   # 1D kernel
            for i in range(len(gain_matrix)):
                calc_max_conductance, calc_relative_reversal_potential = __calc_synaptic_parameters_from_gain__(gain_matrix[i],
                                                                                                                positive_reversal_potential,
                                                                                                                negative_reversal_potential,
                                                                                                                self.params['R'])
                self.params['max_conductance'].append(calc_max_conductance)
                self.params['relative_reversal_potential'].append(calc_relative_reversal_potential)

class SpikingPatternConnection(SpikingConnection):
    def __init__(self,gain_matrix,
                 transmission_delay_matrix,
                 name: str = 'Pattern',
                 R: float = 20.0,
                 positive_reversal_potential: float = 160.0,
                 negative_reversal_potential: float = -80.0,
                 wrap: bool = False,
                 max_frequency: float = 10.0,
                 nonlinearity: float = 0.1):
        """
        Connection pattern between two neural populations (i.e. a kernel)
        :param gain_matrix: Matrix (or vector for 1D kernels) of synaptic gains
        :param name:    Name of this connection type
        :param R:   Voltage range of neural activity (mV)
        :param positive_reversal_potential: Reversal potential for connections with positive gain (mV)
        :param negative_reversal_potential: Reversal potential for connections with negative gain (mV)
        :param wrap:    Flag for if connections should wrap from one end of the population to the other
        :param max_frequency:   Maximum spiking frequency of the network
        """
        max_conductance = []
        relative_reversal_potential = []
        if not isinstance(max_frequency, numbers.Number):
            raise TypeError('Max spiking frequency must be a number (int, float, double, etc.) greater than 0')
        elif max_frequency <= 0:
            raise ValueError('Max spiking frequency must be greater than 0')
        if isinstance(nonlinearity, numbers.Number):
            if (nonlinearity < 1.0) and (nonlinearity > 0.0):
                time_constant = -1/(max_frequency*math.log(nonlinearity))
            else:
                raise ValueError('Nonlinearity coefficient must be between 0 and 1')
        else:
            raise TypeError('Nonlinearity coefficient must be a number (int, float, double, etc.) between 0 and 1')
        super().__init__(max_conductance,relative_reversal_potential,time_constant,transmission_delay_matrix,R,name=name)
        self.params['wrap'] = wrap
        self.params['pattern'] = True

        self.positive_reversal_potential = positive_reversal_potential
        self.negative_reversal_potential = negative_reversal_potential

        if hasattr(gain_matrix[0],'__iter__'): # 2D kernel
            for row in range(len(gain_matrix)):
                cond_values = []
                del_e_values = []
                for col in range(len(gain_matrix[0])):
                    calc_max_conductance,calc_relative_reversal_potential = __calc_spiking_synaptic_parameters_from_gain__(gain_matrix[row][col],
                                                                                                                           positive_reversal_potential,
                                                                                                                           negative_reversal_potential,
                                                                                                                           self.params['R'],
                                                                                                                           time_constant,
                                                                                                                           max_frequency)
                    cond_values.append(calc_max_conductance)
                    del_e_values.append(calc_relative_reversal_potential)
                self.params['max_conductance'].append(cond_values)
                self.params['relative_reversal_potential'].append(del_e_values)
        else:   # 1D kernel
            for i in range(len(gain_matrix)):
                calc_max_conductance, calc_relative_reversal_potential = __calc_synaptic_parameters_from_gain__(gain_matrix[i],
                                                                                                                positive_reversal_potential,
                                                                                                                negative_reversal_potential,
                                                                                                                self.params['R'])
                self.params['max_conductance'].append(calc_max_conductance)
                self.params['relative_reversal_potential'].append(calc_relative_reversal_potential)

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

class SpikingTransmissionSynapse(SpikingSynapse):
    def __init__(self,gain: float = 1.0, name: str = 'Transmit', max_frequency: float = 10.0, nonlinearity: float = 0.1,
                 **kwargs) -> None:
        """
        Spiking version of the transmission synapse.
        :param gain: Transmission frequency gain
        :param name: Name of this preset
        :param max_frequency: Maximum neural spiking frequency
        :param nonlinearity: Degree of nonlinear response of firing rate
        """
        if not isinstance(max_frequency, numbers.Number):
            raise TypeError('Max spiking frequency must be a number (int, float, double, etc.) greater than 0')
        elif max_frequency <= 0:
            raise ValueError('Max spiking frequency must be greater than 0')
        if isinstance(nonlinearity, numbers.Number):
            if (nonlinearity < 1.0) and (nonlinearity > 0.0):
                time_constant = -1/(max_frequency*math.log(nonlinearity))
            else:
                raise ValueError('Nonlinearity coefficient must be between 0 and 1')
        else:
            raise TypeError('Nonlinearity coefficient must be a number (int, float, double, etc.) between 0 and 1')
        super().__init__(time_constant=time_constant,name=name,**kwargs)
        if isinstance(gain,numbers.Number):
            if gain == 0:
                raise ValueError('Gain must be nonzero')
            elif math.copysign(1,gain) != math.copysign(1,self.params['relative_reversal_potential']):    # sign of integration_gain and DeltaE don't match
                raise ValueError('Gain of '+str(gain)+' and Relative Reversal Potential must have the same sign')
            else:
                try:
                    self.params['max_conductance'] = (gain*self.params['R'])/((self.params['relative_reversal_potential']-gain*-self.params['R'])*time_constant*max_frequency)
                except ZeroDivisionError:
                    raise ValueError('Gain of '+str(gain)+' causes division by 0, decrease integration_gain or increase relative_reversal_potential')
                if self.params['max_conductance'] < 0:
                    raise ValueError('Gain of '+str(gain)+' causes max_conductance to be negative, decrease integration_gain or increase relative_reversal_potential')
        else:
            raise TypeError('Gain of '+str(gain)+' must be a number (int, float, double, etc.)')

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
HELPER FUNCTIONS
"""

def __calc_synaptic_parameters_from_gain__(gain, positive_reversal_potential, negative_reversal_potential, R):
    if gain == 0.0:
        return 0.0, 0.0
    else:
        if gain > 0.0:
            relative_reversal_potential = positive_reversal_potential
        else:
            relative_reversal_potential = negative_reversal_potential
        max_conductance = gain * R / (relative_reversal_potential - gain * R)

        return max_conductance, relative_reversal_potential

def __calc_spiking_synaptic_parameters_from_gain__(gain, positive_reversal_potential, negative_reversal_potential, R,
                                                   time_constant, max_frequency):
    if gain == 0.0:
        return 0.0, 0.0
    else:
        if gain > 0.0:
            relative_reversal_potential = positive_reversal_potential
        else:
            relative_reversal_potential = negative_reversal_potential
        max_conductance = gain * R / ((relative_reversal_potential - gain * R)*time_constant*max_frequency)

        return max_conductance, relative_reversal_potential
