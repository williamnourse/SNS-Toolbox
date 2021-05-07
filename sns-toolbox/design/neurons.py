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

from __utilities__ import validColor, setTextColor

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
BASE CLASS
"""

class Neuron:
    def __init__(self, name: str = 'Neuron', color: str = 'white'):
        self.params: Dict[str, Any] = {}
        if validColor(color):
            self.params['color'] = color
        else:
            warnings.warn('WARNING: Specified color is not in the standard SVG set. Defaulting to white.')
        self.params['fontColor'] = setTextColor(self.params['color'])
        self.params['name'] = name

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SPECIFIC MODELS
"""

# Only one of these (for now, or for forever *shrugs*

