"""""""""""""""""
Connection Models
"""""""""""""""""

Neurons, and populations of neurons, are connected in SNS-Toolbox using *connections*. The different versions available
are as follows:

====================
Non-Spiking Synapse:
====================

The most basic form of synaptic connection. The amount of synaptic current :math:`I_{syn}^{ji}` from pre-synaptic neuron
:math:`j` to post-synaptic neuron :math:`i` is

.. math::
    I_{syn}^{ji} = G_{syn}^{ji}(U_j) \cdot \left ( \Delta E_{syn}^{ji} - U_i \right ),

where :math:`\Delta E_{syn}^{ji}` is the relative reversal potential of the synapse, and :math:`G_{syn}^{ji}(U_j)` is
the synaptic conductance as a function of the pre-synaptic neural voltage:

.. math::
    G_{syn}^{ji}(U_j) = max \left ( 0, min \left ( G_{max,non}^{ji} \cdot \frac{U_j}{R}, G_{max,non}^{ji} \right ) \right )

.. image:: images/linear_conductance.png
    :width: 600

