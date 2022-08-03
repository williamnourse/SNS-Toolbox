"""
Functions which are used within the design subpackage, but may not be necessary for the end user.
"""

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
IMPORTS
"""

import warnings

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
