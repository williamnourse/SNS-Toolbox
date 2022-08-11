"""
Neurons are the computational nodes of an SNS, and can be either spiking or non-spiking.
"""

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
IMPORTS
"""

from typing import Dict, Any
import warnings
import numbers

from sns_toolbox.color_utilities import valid_color

import numpy as np
import torch

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
BASE CLASS
"""

class Neuron:
    """
    Parent class of all neurons.

    :param name:                    Name of this neuron preset, default is 'Neuron'.
    :type name:                     str, optional
    :param color:                   Background fill color for the neuron, default is 'white'.
    :type color:                    str, optional
    :param membrane_capacitance:    Neural membrane capacitance, default is 5.0. Units are nanofarads (nF).
    :type membrane_capacitance:     Number, optional
    :param membrane_conductance:    Neural membrane conductance, default is 1.0. Units are microsiemens (uS).
    :type membrane_conductance:     Number, optional
    :param resting_potential:       Neural resting potential, default is 0.0. Units are millivolts (mV).
    :type resting_potential:        Number, optional
    :param bias:                    Internal bias current, default is 0.0. Units are nanoamps (nA).
    :type bias:                     Number, optional
    """
    def __init__(self, name: str = 'Neuron',
                 color: str = 'white',
                 membrane_capacitance: float = 5.0,
                 membrane_conductance: float = 1.0,
                 resting_potential: float = 0.0,
                 bias: float = 0.0) -> None:

        self.params: Dict[str, Any] = {}
        if valid_color(color):
            self.color = color
        else:
            warnings.warn('Specified color is not in the standard SVG set. Defaulting to white.')
            self.color = 'white'
        if isinstance(name,str):
            self.name = name
        else:
            raise TypeError('Neuron name must be a string')
        if isinstance(membrane_capacitance, numbers.Number):
            self.params['membrane_capacitance'] = membrane_capacitance
        else:
            raise TypeError('Membrane capacitance must be a number (int, float, double, etc.)')
        if isinstance(membrane_conductance, numbers.Number):
            self.params['membrane_conductance'] = membrane_conductance
        else:
            raise TypeError('Membrane conductance must be a number (int, float, double, etc.)')
        if isinstance(resting_potential, numbers.Number):
            self.params['resting_potential'] = resting_potential
        else:
            raise TypeError('Resting potential must be a number (int, float, double, etc.)')
        if isinstance(bias,numbers.Number):
            self.params['bias'] = bias
        else:
            raise TypeError('Bias must be a number (int, float, double, etc.')
        self.params['spiking'] = False
        self.params['gated'] = False

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SPECIFIC MODELS
"""

class NonSpikingNeuron(Neuron):
    """
    Classic non-spiking neuron model, whose dynamics are as follows:
    membrane_capacitance*dV/dt = -membrane_conductance*(V - Er) + bias current + synaptic current + external current.
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)


class NonSpikingNeuronWithGatedChannels(NonSpikingNeuron):
    """
    Iion = sum_j[Gj * A_(inf,j)^Pa * Bj^Pb * Cj^Pc * (Ej - V)]
    """
    def __init__(self, g_ion, e_ion,
                 pow_a, k_a, slope_a, e_a,
                 pow_b, k_b, slope_b, e_b, tau_max_b,
                 pow_c, k_c, slope_c, e_c, tau_max_c, **kwargs) -> None:
        super().__init__(**kwargs)

        inputs = [g_ion, e_ion,                         # Channel params
                  pow_a, k_a, slope_a, e_a,             # A gate params
                  pow_b, k_b, slope_b, e_b, tau_max_b,  # B gate params
                  pow_c, k_c, slope_c, e_c, tau_max_c]  # C gate params
        if any(inputs) is False:
            raise ValueError('All channel parameters must have a value')
        if all(len(x) == len(g_ion) for x in inputs) is False:
            raise ValueError('All channel parameters must be the same dimension (len(g_ion) = len(e_ion) = ...)')
        self.params['gated'] = True
        self.params['Gion'] = g_ion
        self.params['Eion'] = e_ion
        self.params['numChannels'] = len(g_ion)
        self.params['paramsA'] = {'pow': pow_a, 'k': k_a, 'slope': slope_a, 'reversal': e_a}
        self.params['paramsB'] = {'pow': pow_b, 'k': k_b, 'slope': slope_b, 'reversal': e_b, 'TauMax': tau_max_b}
        self.params['paramsC'] = {'pow': pow_c, 'k': k_c, 'slope': slope_c, 'reversal': e_c, 'TauMax': tau_max_c}


class NonSpikingNeuronWithPersistentSodiumChannel(NonSpikingNeuronWithGatedChannels):
    """
    Iion = sum_j[Gj * m_(inf,j)^Pm * hj^Ph * (Ej - U)]
    """
    def __init__(self, g_ion=None, e_ion=None,
                 k_m=None, slope_m=None, e_m=None,
                 k_h=None, slope_h=None, e_h=None, tau_max_h=None,**kwargs):
        if g_ion is None:
            g_ion = np.array([1.0485070729908987])
        if e_ion is None:
            e_ion = np.array([110])

        if k_m is None:
            k_m = np.array([1])
        if slope_m is None:
            slope_m = np.array([0.05])
        if e_m is None:
            e_m = np.array([20])

        if k_h is None:
            k_h = np.array([0.5])
        if slope_h is None:
            slope_h = np.array([-0.05])
        if e_h is None:
            e_h = np.array([0])
        if tau_max_h is None:
            tau_max_h = np.array([300])

        inputs = [g_ion, e_ion,                         # Channel params
                  k_m, slope_m, e_m,                    # A gate params
                  k_h, slope_h, e_h, tau_max_h]         # B gate params
        if all(len(x) == len(g_ion) for x in inputs) is False:
            raise ValueError('All channel parameters must be the same dimension (len(g_ion) = len(e_ion) = ...)')
        num_channels = len(g_ion)
        if isinstance(g_ion, torch.Tensor):
            device = g_ion.device
            pow_m = torch.tensor([1],device=device)
            pow_h = torch.tensor([1],device=device)
            pow_c = torch.zeros(num_channels,device=device)
            k_c = torch.zeros(num_channels,device=device) + 1
            slope_c = torch.zeros(num_channels,device=device)
            e_c = torch.zeros(num_channels,device=device)
            tau_max_c = torch.zeros(num_channels,device=device) + 1
        else:
            pow_m = np.array([1])
            pow_h = np.array([1])
            pow_c = np.zeros(num_channels)
            k_c = np.zeros(num_channels) + 1
            slope_c = np.zeros(num_channels)
            e_c = np.zeros(num_channels)
            tau_max_c = np.zeros(num_channels) + 1

        super().__init__(g_ion=g_ion, e_ion=e_ion,
                         pow_a=pow_m, k_a=k_m, slope_a=slope_m, e_a=e_m,
                         pow_b=pow_h, k_b=k_h, slope_b=slope_h, e_b=e_h, tau_max_b=tau_max_h,
                         pow_c=pow_c, k_c=k_c, slope_c=slope_c, e_c=e_c, tau_max_c=tau_max_c, **kwargs)


class SpikingNeuron(Neuron):
    """
    Generalized leaky integrate-and-fire neuron model, whose dynamics are as follows:
    membrane_capacitance*dV/dt = -membrane_conductance*(V-Er) + bias current + synaptic current + external current;
    threshold_time_constant*dTheta/dt = -Theta + threshold_initial_value + threshold_proportionality_constant*V;
    if V > Theta, V->Er.

    :param threshold_time_constant: Rate that the firing threshold moves to the baseline value, default is 5.0. Units
        are milliseconds (ms).
    :type threshold_time_constant:  Number, optional
    :param threshold_initial_value: Baseline value of the firing threshold, default is 1.0. Units are millivolts (mV).
    :type threshold_initial_value:  Number, optional
    :param threshold_proportionality_constant:  Constant which determines spiking behavior.
        In response to constant stimulus, negative values cause the firing rate to decrease, positive values cause the
        rate to increase, and zero causes the rate to remain constant. Default is 0.0.
    :param threshold_proportionality constant:  Number, optional
    """
    def __init__(self, threshold_time_constant: float = 5.0,
                 threshold_initial_value: float = 1.0,
                 threshold_proportionality_constant: float = 0.0,
                 **kwargs) -> None:

        super().__init__(**kwargs)
        self.params['threshold_time_constant'] = threshold_time_constant
        self.params['threshold_initial_value'] = threshold_initial_value
        self.params['threshold_proportionality_constant'] = threshold_proportionality_constant
        self.params['spiking'] = True
