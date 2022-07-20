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

from sns_toolbox.design.design_utilities import valid_color, set_text_color

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
    :param membrane_rest_potential: Neural Resting Potential, default to -60 (mV)
    :type membrane_rest_potential:  Number, optional
    :param pre_synaptic_saturation: Sets the membrane potential of the pre-synaptic neuron at which synaptic activity saturates, default to -40 (mV)
    :type pre_synaptic_saturation:  Number, optional
    :param pre_synaptic_threshold: Sets the threshold membrane potential of the pre-synaptic neuron for synaptic activity , default to -60 (mV)
    :type pre_synaptic_threshold:  Number, optional
    :param bias:                    Internal bias current, default is 0.0. Units are nanoamps (nA).
    :type bias:                     Number, optional
    """
    def __init__(self, name: str = 'Neuron',
                 color: str = 'white',
                 membrane_capacitance: float = 5.0,
                 membrane_conductance: float = 1.0,
                 membrane_rest_potential: float = -60.0,
                 pre_synaptic_saturation: float = -40.0,
                 pre_synaptic_threshold: float = -60.0,
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
            raise TypeError('Membrane conductance must be a number (int, float, double, etc.')
        if isinstance(membrane_rest_potential, numbers.Number):
            self.params['membrane_rest_potential'] = membrane_rest_potential
        else:
            raise TypeError('Membrane rest potential must be a number (int, float, double, etc.')
        if isinstance(pre_synaptic_saturation, numbers.Number) and isinstance(pre_synaptic_threshold, numbers.Number):
            self.params['R'] = pre_synaptic_saturation - pre_synaptic_threshold
            self.params['pre_synaptic_threshold'] = pre_synaptic_threshold
            self.params['[pre_synaptic_saturation'] = pre_synaptic_saturation
        else:
            raise TypeError('pre_synaptics must be numbers (int, float, double, etc.')
        if isinstance(bias,numbers.Number):
            self.params['bias'] = bias
        else:
            raise TypeError('Bias must be a number (int, float, double, etc.')

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SPECIFIC MODELS
"""

class NonSpikingNeuron(Neuron):
    """
    Classic non-spiking neuron model, whose dynamics are as follows:
    membrane_capacitance*dV/dt = -membrane_conductance*(V - membrane_rest_potential) + bias current + synaptic current + external current.
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

class SpikingNeuron(Neuron):
    """
    Generalized leaky integrate-and-fire neuron model, whose dynamics are as follows:
    membrane_capacitance*dU/dt = -membrane_conductance*u + bias current + synaptic current + external current;
    threshold_time_constant*dTheta/dt = -Theta + threshold_initial_value + threshold_proportionality_constant*u;
    if u > Theta, u->0.

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
