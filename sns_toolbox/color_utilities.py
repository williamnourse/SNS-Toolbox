"""
Utility functions for dealing with network colors.
"""
"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
IMPORTS
"""

import warnings
from matplotlib import pyplot as plt

SETOFVALIDCOLORS = {'aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'black', 'blanchedalmond',
          'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral',
          'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray',
          'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred',
          'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet',
          'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen',
          'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'gray', 'grey', 'green', 'greenyellow', 'honeydew',
          'hotpink', 'indianred', 'indigo',	'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen',	'lemonchiffon',
          'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey',
          'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey',
          'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon', 'mediumaquamarine',
                    'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen',
                    'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite',
                    'navy', 'oldlace', 'olive', 'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen',
                    'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple',
                    'red', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna',
                    'silver', 'skyblue', 'slateblue', 'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'teal',
                    'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'white', 'whitesmoke', 'yellow', 'yellowgreen'}
"""
Set of standard SVG color names.
"""

COLORSWHITETEXT = {'black', 'blue', 'blueviolet', 'brown', 'darkblue', 'darkmagenta', 'darkolivegreen', 'darkred',
          'darkslateblue', 'darkslategray', 'darkslategrey', 'darkviolet', 'dimgray', 'dimgrey', 'indigo', 'maroon',
                   'mediumblue', 'midnightblue', 'navy', 'olive', 'purple', 'saddlebrown', 'teal'}
"""
Set of background colors which work better with white text for improved visibility.
"""


def valid_color(color: str = None) -> bool:
    """
    Check if a given color is within the standard svg set.

    :param color: Desired color, default is None.
    :type color: str, optional
    :return: True if it is a valid color, false if not.
    :rtype: bool
    """
    if isinstance(color, str):
        if color in SETOFVALIDCOLORS:
            return True
        else:
            return False
    else:
        return False

def set_text_color(color: str = None) -> str:
    """
    Set the text color to white or black given a valid color string.

    :param color: Background color, default is None.
    :type color: str, optional
    :return: Text color based on background color.
    :rtype: str
    """
    if isinstance(color, str):
        if color in COLORSWHITETEXT:
            return 'white'
        else:
            return 'black'
    else:
        warnings.warn('Specified background color is not a string. Defaulting to black text color')
        return 'black'

def spike_raster_plot(t,data,colors=None,offset=0) -> None:
    """
    Plot spike rasters of spiking data.

    :param t:   vector of timesteps.
    :type t:    List, np.ndarray, or torch.tensor
    :param data:    2D vector of spiking data. Each row corresponds to a different neuron.
    :type data:     np.ndarray or torch.tensor
    :param colors:  List of colors to plot each neuron, default is every neuron is blue.
    :type colors:   List of str, optional
    :param offset:  Constant vertical offset for all spikes, default is 0.
    :type offset:   Number, optional
    :return:        None
    :rtype:         N/A
    """
    if colors is None:
        colors = ['blue']
    if data.ndim > 1:
        for neuron in range(len(data)):
            spike_locs = []
            for step in range(len(t)):
                if data[neuron][step] > 0:
                    spike_locs.append(t[step])
            if len(colors) == 1:
                plt.eventplot(spike_locs,lineoffsets=neuron+1+offset, colors=colors[0],linelengths=0.8)
            else:
                plt.eventplot(spike_locs, lineoffsets=neuron + 1+offset, colors=colors[neuron], linelengths=0.8)
    else:
        spike_locs = []
        for step in range(len(t)):
            if data[step] > 0:
                spike_locs.append(t[step])
        plt.eventplot(spike_locs, lineoffsets=1+offset, colors=colors[0], linelengths=0.8)
