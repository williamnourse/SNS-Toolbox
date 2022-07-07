"""
The mechanism for defining a neuron model which can be simulated in the SNS Toolbox
William Nourse
May 7, 2021
Execute Order 66
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
    def __init__(self, name: str = 'Neuron',
                 color: str = 'white',
                 membrane_capacitance: float = 5.0,
                 membrane_conductance: float = 1.0,
                 bias: float = 0.0) -> None:
        """
        Constructor for base Neuron class
        :param name:    Name of this neuron preset
        :param color:   Background fill color for the neuron, taken from the standard SVG colors
        :param membrane_capacitance: (nF)
        :param membrane_conductance: (uS)
        :param bias:                Bias current (nA)
        """
        # TODO: Type checking
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
        if isinstance(bias,numbers.Number):
            self.params['bias'] = bias
        else:
            raise TypeError('Bias must be a number (int, float, double, etc.')

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SPECIFIC MODELS
"""

class NonSpikingNeuron(Neuron):
    def __init__(self, **kwargs) -> None:
        """
        Classic non-spiking neuron model, whose dynamics are as follows:
        membrane_capacitance*dU/dt = -membrane_conductance*u + bias + synaptic + external
        """
        super().__init__(**kwargs)

class SpikingNeuron(Neuron):
    def __init__(self, threshold_time_constant: float = 5.0,
                 threshold_initial_value: float = 1.0,
                 threshold_proportionality_constant: float = 0.0,
                 **kwargs) -> None:
        """
        Generalized leaky integrate-and-fire neuron model, whose dynamics are as follows:
        membrane_capacitance*dU/dt = -membrane_conductance*u + bias + synaptic + external
        threshold_time_constant*dTheta/dt = -Theta + threshold_initial_value + threshold_proportionality_constant*u
        if u > Theta, u->0
        """
        super().__init__(**kwargs)
        self.params['threshold_time_constant'] = threshold_time_constant
        self.params['threshold_initial_value'] = threshold_initial_value
        self.params['threshold_proportionality_constant'] = threshold_proportionality_constant
