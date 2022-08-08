""""""""""""""
Neural Models
""""""""""""""

A variety of neuron models are available to use within SNS-Toolbox, each with different dynamics.

==================
Non-Spiking Neuron:
==================

Non-spiking neurons are simulated as leaky integrators, where the membrane depolarization above rest (:math:`U`) behaves
according to the differential equation
.. math::
    C_{mem} = \frac{dU}{dt} = -G_{mem}\cdot U + \sum I_{syn} + I_{bias} + I_{app},

where :math:`C_m` is the membrane capacitance, :math:`G_{mem}` is the membrane leak conductance, :math:`I_{bias}` is a
constant offset current, :math:`I_{app}` is an external applied current, and :math:`I_{syn}` is the current induced by
an incoming conductance-based or electrical synapse. This neuron can be implemented using
`sns_toolbox.design.neurons.NonSpikingNeuron`.