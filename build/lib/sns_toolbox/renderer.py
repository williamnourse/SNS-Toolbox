"""
Take in a network, and render its structure using the package 'graphviz'.
"""
from sns_toolbox.networks import Network
from sns_toolbox.color_utilities import set_text_color

import warnings

graphviz_available = True
try:    # see if graphviz is available
    from graphviz import Digraph
except ImportError:
    warnings.warn('Warning: graphviz package not found, network rendering is disabled')
    graphviz_available = False

try:    # check if we're in an ipython environment
    shell = get_ipython().__class__.__name__
    in_ipython = True
except NameError:
    in_ipython = False

def render(net: Network, view=True, save=False, filename=None, img_format='png'):
    if not isinstance(net, Network):
        raise TypeError('Invalid Network object')
    if filename is None:
        filename = net.params['name']
    if graphviz_available:
        graph = Digraph(filename=filename)
        """Add the populations"""
        for i in range(len(net.populations)):
            pop = net.populations[i]
            color_cell = pop['color']   # get cell color
            pop_size = pop['number']    # get pop size
            name = pop['name']  # get cell name
            color_font = set_text_color(color_cell) # calc font color
            pop_shape = pop['shape']
            # set node shape
            if pop_size > 1:    # population has multiple neurons
                style = 'filled, rounded'
                if len(pop_shape) > 1:
                    shape = 'box3d' # population is 2d
                else:
                    shape = 'rect'
            else:   # population is 1 neuron
                shape = 'ellipse'
                style = 'filled'
            graph.node(str(i), label=name, style=style, shape=shape, fillcolor=color_cell, fontcolor=color_font)

        """Add the inputs"""
        for i in range(len(net.inputs)):
            inp = net.inputs[i]
            name = inp['name']
            size = inp['size']
            color_cell = inp['color']
            color_font = set_text_color(color_cell)
            destination = inp['destination']
            if size > 1:
                style = 'filled, diagonals'
            else:
                style = 'filled'
            graph.node('In'+str(i), label=name, style=style, shape='invhouse', fillcolor=color_cell, fontcolor=color_font)
            graph.edge('In'+str(i), str(destination))

        """Add the outputs"""
        for i in range(len(net.outputs)):
            out = net.outputs[i]
            name = out['name']
            color_cell = out['color']
            color_font = set_text_color(color_cell)
            source = out['source']
            spiking = out['spiking']
            if spiking:
                shape = 'triangle'
            else:
                shape = 'house'
            if net.populations[source]['number'] > 1:
                style = 'filled, diagonals'
            else:
                style = 'filled'
            graph.node('Out' + str(i), label=name, style=style, shape=shape, fillcolor=color_cell,
                       fontcolor=color_font)
            graph.edge(str(source), 'Out' + str(i))

        """Add the connections"""
        for i in range(len(net.connections)):
            conn = net.connections[i]
            label = conn['name']
            source = conn['source']
            destination = conn['destination']
            params = conn['params']
            view_label = conn['view']
            if params['pattern']:   # pattern connection
                style = 'vee'
                direction = 'forward'
            elif params['electrical']:  # electrical synapese
                style = 'odiamond'
                if params['rectified']: # rectified
                    direction = 'forward'
                else:   # not rectified
                    direction = 'both'
            else:   # chemical synapse
                direction = 'forward'
                if params['reversal_potential'] > 0:   # excitatory
                    style = 'invempty'
                elif params['reversal_potential'] < 0: # inhibitory
                    style = 'dot'
                else:   # modulatory
                    style = 'odot'
            if view_label:
                graph.edge(str(source), str(destination), dir=direction, arrowhead=style, label=label, arrowtail=style)
            else:
                graph.edge(str(source), str(destination), dir=direction, arrowhead=style, arrowtail=style)

        """Display and/or save the network graph"""
        graph.format = img_format
        if save:
            graph.render(view=False,cleanup=True)
        if view:
            if in_ipython:
                return graph
            else:
                graph.render(view=True, cleanup=True)
    else:
        warnings.warn('Warning: graphviz package not found, network rendering is disabled')
