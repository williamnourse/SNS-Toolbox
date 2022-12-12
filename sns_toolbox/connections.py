"""
Connections are the mechanisms for data transmission between neurons. They can either define an individual conductance-
based synapse, or a pattern of synapses which is tiled between two populations.
"""
import numpy as np

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
IMPORTS
"""

from typing import Dict, Any
import numbers
import math

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
BASE CLASSES
"""
class Connection:
    """
    Base class of all connections. Initializes a dictionary of parameters which is modified by classes which inherit
    from it.

    :param max_conductance: All connections have a maximum synaptic conductance. It can be a single value or a matrix,
        but it must be defined.
    :type max_conductance: Number, np.ndarray, or torch.Tensor
    :param name: Name of this connection preset, defaults to 'Connection'.
    :type name: str, optional
    """
    def __init__(self, max_conductance, name: str = 'Connection'):
        self.params: Dict[str, Any] = {}
        self.params['spiking'] = False
        self.params['pattern'] = False
        self.params['electrical'] = False
        self.params['max_conductance'] = max_conductance
        if isinstance(name, str):
            self.params['name'] = name
        else:
            raise TypeError('Name should be a string')

class ElectricalSynapse(Connection):
    def __init__(self, conductance, name: str = 'Electrical Synapse', rect: bool = False) -> None:
        super().__init__(conductance, name)
        self.params['electrical'] = True
        self.params['rectified'] = rect

class NonSpikingConnection(Connection):
    """
    Base class of all non-spiking connections. Initializes a dictionary of parameters which is modified by classes
    which inherit from it.

    :param max_conductance: All connections have a maximum synaptic conductance. It can be a single value or a matrix,
        but it must be defined.
    :type max_conductance: Number, np.ndarray, or torch.Tensor
    :param reversal_potential: All chemical connections have a relative synaptic reversal potential. It can be
        a single value or a matrix, but it must be defined.
    :type reversal_potential: Number, np.ndarray, or torch.Tensor
    :param e_lo: Synaptic activation threshold.
    :type e_lo: Number, np.ndarray, or torch.Tensor
    :param e_hi: Synaptic maximum activation limit.
    :type e_hi: Number, np.ndarray, or torch.Tensor
    :param name: Name of this connection preset, defaults to 'Non-Spiking Connection'.
    :type name: str, optional
    """
    def __init__(self, max_conductance, reversal_potential, e_lo, e_hi, name: str = 'Non-Spiking Connection') -> None:
        super().__init__(max_conductance, name)
        self.params['reversal_potential'] = reversal_potential
        self.params['spiking'] = False
        self.params['e_lo'] = e_lo
        self.params['e_hi'] = e_hi

class SpikingConnection(Connection):
    """
    Base class of all spiking connections. Initializes a dictionary of parameters which is modified by classes
    which inherit from it.

    :param max_conductance: All connections have a maximum synaptic conductance. It can be a single value or a matrix,
        but it must be defined.
    :type max_conductance: Number, np.ndarray, or torch.Tensor
    :param reversal_potential: All chemical connections have a relative synaptic reversal potential. It can be
        a single value or a matrix, but it must be defined.
    :type reversal_potential: Number, np.ndarray, or torch.Tensor
    :param name: Name of this connection preset, defaults to 'Spiking Connection'.
    :type name: str, optional
    """
    def __init__(self, max_conductance, reversal_potential, time_constant, transmission_delay,
                 name: str = 'Spiking Connection') -> None:
        super().__init__(max_conductance, name)
        self.params['spiking'] = True
        self.params['reversal_potential'] = reversal_potential
        self.params['synapticTimeConstant'] = time_constant
        self.params['transmissionDelay'] = transmission_delay

class NonSpikingSynapse(NonSpikingConnection):
    """
    An individual non-spiking synapse, where the conductance is defined as Conductance = max_conductance * max(0,
    min(1, Upre/R)), and the synaptic current is i_syn = Conductance*(reversal_potential - Upost).

    :param max_conductance: Maximum synaptic conductance, defaults to 1.0. Units are micro-siemens (uS).
    :type max_conductance: float, optional
    :param reversal_potential: Synaptic reversal potential, defaults to 40.0. Units are millivolts (mV).
    :type reversal_potential: float, optional
    :param e_lo: Synaptic activation threshold.
    :type e_lo: float
    :param e_hi: Synaptic maximum activation limit.
    :type e_hi: float
    """
    def __init__(self, max_conductance: float = 1.0,
                 reversal_potential: float = 40.0,
                 e_lo: float = 0.0,
                 e_hi: float = 20.0,
                 **kwargs: Any) -> None:
        if isinstance(max_conductance, numbers.Number):
            if max_conductance < 0:
                raise ValueError('max_conductance (gMax) must non-negative')
        else:
            raise TypeError('max_conductance (gMax) must be a number (int, float, double, etc.')
        if not isinstance(reversal_potential, numbers.Number):
            raise TypeError('reversal_potential (deltaEsyn) must be a number (int, float, double, etc.')
        super().__init__(max_conductance, reversal_potential, e_lo, e_hi, **kwargs)  # Call to constructor of parent class
        self.params['pattern'] = False

class SpikingSynapse(SpikingConnection):
    """
    An individual spiking synapse, where the conductance is reset to max_conductance whenever the pre-synaptic
    neuron spikes, and otherwise decays to zero according to the time constant. Synaptic current
    is i_syn = Conductance*(reversal_potential - Upost). Synaptic propagation can be delayed by a set number
    of timesteps.

    :param max_conductance: Maximum synaptic conductance, defaults to 1.0. Units are micro-siemens (uS).
    :type max_conductance: float, optional
    :param reversal_potential: Synaptic reversal potential, defaults to 194.0. Units are millivolts (mV).
    :type reversal_potential: float, optional
    :param time_constant: Time constant of synaptic decay, defaults to 1.0. Units are milliseconds (ms).
    :type time_constant: float, optional
    :param transmission_delay: Number of timesteps to delay synaptic activity, defaults to 0. Units are timesteps (dt).
    :type transmission_delay: int, optional
    """
    def __init__(self, max_conductance: float = 1.0,
                 reversal_potential: float = 194.0,
                 time_constant: float = 1.0,
                 transmission_delay: int = 0,
                 **kwargs: Any) -> None:
        if isinstance(max_conductance, numbers.Number):
            if max_conductance <= 0:
                raise ValueError('max_conductance (gMax) must be greater than 0')
        else:
            raise TypeError('max_conductance (gMax) must be a number (int, float, double, etc.')
        if not isinstance(reversal_potential, numbers.Number):
            raise TypeError('reversal_potential (deltaEsyn) must be a number (int, float, double, etc.')
        super().__init__(max_conductance, reversal_potential, time_constant, transmission_delay, **kwargs)  # Call to constructor of parent class
        self.params['pattern'] = False
        if isinstance(time_constant, numbers.Number):
            if time_constant > 0:
                self.params['synapticTimeConstant'] = time_constant
            else:
                raise ValueError('Synaptic time constant (tauS) must be greater than 0')
        else:
            raise TypeError('Synaptic time constant (tauS) must be a number (int, float, double, etc.) greater than 0')

        if isinstance(transmission_delay, int):
            if transmission_delay >= 0:
                self.params['synapticTransmissionDelay'] = transmission_delay
            else:
                raise ValueError('Synaptic transmission delay must be greater than or equal to zero')
        else:
            raise TypeError('Synaptic transmission delay must be an integer')

class NonSpikingPatternConnection(NonSpikingConnection):
    """
    A pattern of non-spiking synapses, with kernel matrices representing the maximum conductance and reversal potential
    of each synapse in the pattern.

    :param max_conductance_kernel: Kernel matrix of conductance values. Units are micro-siemens (uS).
    :type max_conductance_kernel: np.ndarray or torch.Tensor
    :param reversal_potential_kernel: Kernel matrix of reversal potential values. Units are millivolts (mV).
    :type reversal_potential_kernel: np.ndarray or torch.Tensor
    :param e_lo_kernel: Synaptic activation threshold kernel matrix. Units are millivolts (mV)
    :type e_lo_kernel: np.ndarray, or torch.Tensor
    :param e_hi_kernel: Synaptic maximum activation limit kernel matrix. Units are millivolts (mV)
    :type e_hi_kernel: np.ndarray, or torch.Tensor
    """
    def __init__(self, max_conductance_kernel, reversal_potential_kernel, e_lo_kernel, e_hi_kernel, **kwargs: Any) -> None:
        if max_conductance_kernel.shape != reversal_potential_kernel.shape:
            raise ValueError('Max Conductance and Relative Reversal Potential must be matrices of the same shape')
        if np.any(max_conductance_kernel < 0):
            raise ValueError('Max Conductance values must be non-negative')
        super().__init__(max_conductance_kernel, reversal_potential_kernel, e_lo_kernel, e_hi_kernel, **kwargs)  # Call to constructor of parent class
        self.params['pattern'] = True

class SpikingPatternConnection(SpikingConnection):
    """
    A pattern of spiking synapses, with kernel matrices representing the maximum conductance, reversal potential, time
    constant, and transmission delay of each synapse in the pattern.

    :param max_conductance_kernel: Kernel matrix of conductance values. Units are micro-siemens (uS).
    :type max_conductance_kernel: np.ndarray or torch.tensor
    :param reversal_potential_kernel: Kernel matrix of reversal potential values. Units are millivolts (mV).
    :type reversal_potential_kernel: np.ndarray or torch.tensor
    :param time_constant_kernel: Kernel matrix of time constant values. Units are milliseconds (ms).
    :type time_constant_kernel: np.ndarray or torch.tensor
    :param transmission_delay_kernel: Kernel matrix of transmission delays. Units are timesteps (dt).
    :type transmission_delay_kernel: np.ndarray or torch.tensor
    """
    def __init__(self, max_conductance_kernel, reversal_potential_kernel, time_constant_kernel,
                 transmission_delay_kernel, **kwargs: Any) -> None:
        if (max_conductance_kernel.shape != reversal_potential_kernel.shape) or (
                max_conductance_kernel.shape != time_constant_kernel.shape) or (
                max_conductance_kernel.shape != transmission_delay_kernel.shape):
            raise ValueError('Max Conductance, Relative Reversal Potential, Time Constant, and Transmission Delay must be matrices of the same shape')
        if np.any(max_conductance_kernel < 0):
            raise ValueError('Max Conductance values must be non-negative')
        if np.any(time_constant_kernel <= 0):
            raise ValueError('Time constant values must be greater than 0 ms')
        if np.any(transmission_delay_kernel < 0):
            raise ValueError('Transmission delays must be non-negative')
        super().__init__(max_conductance_kernel, reversal_potential_kernel, time_constant_kernel,
                         transmission_delay_kernel, **kwargs)  # Call to constructor of parent class
        self.params['pattern'] = True

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SPECIFIC MODELS
"""

class NonSpikingTransmissionSynapse(NonSpikingSynapse):
    """
    A non-spiking transmission synapse, where (given some integration_gain) the maximum conductance is
    max_conductance = (integration_gain*R)/(reversal_potential - integration_gain*R).

    :param gain: Transmission integration gain, defaults to 1.0.
    :type gain: Number, optional
    :param name: Name of this synapse preset, defaults to 'Transmit'
    :type name: str, optional
    :param R: Range of neural voltage activity, defaults to 20.0. Units are millivolts (mV).
    :type R: Number
    """
    def __init__(self, gain: float = 1.0,
                 name: str = 'Transmit',
                 R: float = 20.0,
                 **kwargs) -> None:
        super().__init__(name=name, **kwargs)  # Call to constructor of parent class
        if isinstance(gain,numbers.Number): # Gain error handling
            if gain == 0:
                raise ValueError('Gain must be nonzero')
            elif math.copysign(1,gain) != math.copysign(1,self.params['reversal_potential']):    # sign of integration_gain and DeltaE don't match
                raise ValueError('Gain of '+str(gain)+' and Relative Reversal Potential must have the same sign')
            else:
                try:
                    self.params['max_conductance'] = (gain*R)/(self.params['reversal_potential']-gain*R)
                except ZeroDivisionError:
                    raise ValueError('Gain of '+str(gain)+' causes division by 0, decrease integration_gain or increase reversal_potential')
                if self.params['max_conductance'] < 0:
                    raise ValueError('Gain of '+str(gain)+' causes max_conductance to be negative, decrease integration_gain or increase reversal_potential_kernel')
        else:
            raise TypeError('Gain must be a number (int, float, double, etc.)')
        if isinstance(R, numbers.Number):   # R error handling
            if R <= 0:
                raise ValueError('R must be greater than zero')
            else:
                self.params['e_hi'] = self.params['e_lo'] + R
        else:
            raise TypeError('R must be a number (int, float, double, etc.)')

class NonSpikingModulationSynapse(NonSpikingSynapse):
    """
    A non-spiking modulation synapse, where the reversal_potential is set to 0.

    :param ratio: The desired ratio Upost/Upre when Upre is at max activity (R mV).
    :type ratio: Number
    :param name: Name of this synapse preset, defaults to 'Modulate'.
    :type name: str, optional
    """
    def __init__(self,ratio, name: str = 'Modulate', **kwargs) -> None:
        super().__init__(name=name,**kwargs)
        self.params['reversal_potential'] = 0.0
        self.params['e_lo'] = 0.0
        if isinstance(ratio, numbers.Number):   # ratio error handling
            if ratio <= 0:
                raise ValueError('Ratio must be greater than zero')
        else:
            raise ValueError('Ratio must be a number (int, float, double, etc.)')
        self.params['max_conductance'] = 1/ratio - 1

class SpikingTransmissionSynapse(SpikingSynapse):
    """
    A spiking version of the non-spiking transmission synapse.

    :param gain: Transmission frequency gain, defaults to 1.0.
    :type gain: Number, optional
    :param name: Name of this preset, defaults to 'Transmit'.
    :type name: str, optional
    :param max_frequency: Maximum spiking frequency, defaults to 10.0. Units are kilohertz (kHz).
    :type max_frequency: Number, optional
    :param non_linearity: A constant between 0 and 1 which limits the synaptic non-linearity. Values closer to 0 improve
        linearity.
    :type non_linearity: Number, optional
    """
    def __init__(self, gain: float = 1.0, name: str = 'Transmit', max_frequency: float = 10.0,
                 non_linearity: float = 0.1, **kwargs) -> None:
        if not isinstance(max_frequency, numbers.Number):
            raise TypeError('Max spiking frequency must be a number (int, float, double, etc.) greater than 0')
        elif max_frequency <= 0:
            raise ValueError('Max spiking frequency must be greater than 0')
        if isinstance(non_linearity, numbers.Number):
            if (non_linearity < 1.0) and (non_linearity > 0.0):
                time_constant = -1/(max_frequency * math.log(non_linearity))
            else:
                raise ValueError('Nonlinearity coefficient must be between 0 and 1')
        else:
            raise TypeError('Nonlinearity coefficient must be a number (int, float, double, etc.) between 0 and 1')
        super().__init__(time_constant=time_constant,name=name,**kwargs)
        if isinstance(gain,numbers.Number):
            if gain == 0:
                raise ValueError('Gain must be nonzero')
            elif math.copysign(1,gain) != math.copysign(1,self.params['reversal_potential']):    # sign of integration_gain and DeltaE don't match
                raise ValueError('Gain of '+str(gain)+' and Relative Reversal Potential must have the same sign')
            else:
                try:
                    self.params['max_conductance'] = (gain*self.params['R'])/((self.params['reversal_potential']-gain*-self.params['R'])*time_constant*max_frequency)
                except ZeroDivisionError:
                    raise ValueError('Gain of '+str(gain)+' causes division by 0, decrease integration_gain or increase reversal_potential')
                if self.params['max_conductance'] < 0:
                    raise ValueError('Gain of '+str(gain)+' causes max_conductance to be negative, decrease integration_gain or increase reversal_potential')
        else:
            raise TypeError('Gain of '+str(gain)+' must be a number (int, float, double, etc.)')

"""
Classes and functions for an easier pattern connection, just based on gains. Not fully implemented yet.
"""
# class NonSpikingGainPatternConnection(NonSpikingConnection):
#     def __init__(self,gain_matrix,
#                  name: str = 'Pattern',
#                  R: float = 20.0,
#                  positive_reversal_potential: float = 160.0,
#                  negative_reversal_potential: float = -80.0,
#                  wrap: bool = False):
#         """
#         Connection pattern between two neural populations (i.e. a kernel)
#         :param gain_matrix: Matrix (or vector for 1D kernels) of synaptic gains
#         :param name:    Name of this connection type
#         :param R:   Voltage range of neural activity (mV)
#         :param positive_reversal_potential: Reversal potential for connections with positive gain (mV)
#         :param negative_reversal_potential: Reversal potential for connections with negative gain (mV)
#         :param wrap:    Flag for if connections should wrap from one end of the population to the other
#         """
#         max_conductance = []
#         reversal_potential = []
#         super().__init__(max_conductance,reversal_potential,name=name)
#         self.params['wrap'] = wrap
#         self.params['R'] = R
#         self.params['pattern'] = True
#
#         self.positive_reversal_potential = positive_reversal_potential
#         self.negative_reversal_potential = negative_reversal_potential
#
#         if hasattr(gain_matrix[0],'__iter__'): # 2D kernel
#             for row in range(len(gain_matrix)):
#                 cond_values = []
#                 del_e_values = []
#                 for col in range(len(gain_matrix[0])):
#                     calc_max_conductance,calc_relative_reversal_potential = __calc_synaptic_parameters_from_gain__(gain_matrix[row][col],
#                                                                                                                    positive_reversal_potential,
#                                                                                                                    negative_reversal_potential,
#                                                                                                                    self.params['R'])
#                     cond_values.append(calc_max_conductance)
#                     del_e_values.append(calc_relative_reversal_potential)
#                 self.params['max_conductance_kernel'].append(cond_values)
#                 self.params['reversal_potential_kernel'].append(del_e_values)
#         else:   # 1D kernel
#             for i in range(len(gain_matrix)):
#                 calc_max_conductance, calc_relative_reversal_potential = __calc_synaptic_parameters_from_gain__(gain_matrix[i],
#                                                                                                                 positive_reversal_potential,
#                                                                                                                 negative_reversal_potential,
#                                                                                                                 self.params['R'])
#                 self.params['max_conductance_kernel'].append(calc_max_conductance)
#                 self.params['reversal_potential_kernel'].append(calc_relative_reversal_potential)

# def __calc_synaptic_parameters_from_gain__(gain, positive_reversal_potential, negative_reversal_potential, R):
#     if gain == 0.0:
#         return 0.0, 0.0
#     else:
#         if gain > 0.0:
#             reversal_potential = positive_reversal_potential
#         else:
#             reversal_potential = negative_reversal_potential
#         max_conductance = gain * R / (reversal_potential - gain * R)
#
#         return max_conductance, reversal_potential
#
# def __calc_spiking_synaptic_parameters_from_gain__(gain, positive_reversal_potential, negative_reversal_potential, R,
#                                                    time_constant, max_frequency):
#     if gain == 0.0:
#         return 0.0, 0.0
#     else:
#         if gain > 0.0:
#             reversal_potential = positive_reversal_potential
#         else:
#             reversal_potential = negative_reversal_potential
#         max_conductance = gain * R / ((reversal_potential - gain * R)*time_constant*max_frequency)
#
#         return max_conductance, reversal_potential
