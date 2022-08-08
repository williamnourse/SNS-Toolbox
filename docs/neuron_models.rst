""""""""""""""
Neural Models
""""""""""""""

A variety of neuron models are available to use within SNS-Toolbox, each with different dynamics.

===================
Non-Spiking Neuron:
===================

Non-spiking neurons are simulated as leaky integrators, where the membrane depolarization above rest (:math:`U`) behaves
according to the differential equation

.. math::
    C_m \cdot \frac{dU}{dt} = -G_m\cdot U + \sum I_{syn} + I_{bias} + I_{app},

where :math:`C_m` is the membrane capacitance, :math:`G_m` is the membrane leak conductance, :math:`I_{bias}` is a
constant offset current, :math:`I_{app}` is an external applied current, and :math:`I_{syn}` is the current induced by
an incoming conductance-based or electrical synapse.

Default values are as follows:

- :math:`C_m = 5 nF`
- :math:`G_m = 1 \mu S`

This neuron can be implemented using
`sns_toolbox.design.neurons.NonSpikingNeuron
<https://sns-toolbox.readthedocs.io/en/latest/autoapi/sns_toolbox/design/neurons/index.html#sns_toolbox.design.neurons.NonSpikingNeuron>`_.

===============
Spiking Neuron:
===============

Spiking neurons have similar dynamics to the classic non-spiking neuron, with an additional dynamic variable
(:math:`\theta`) that acts as a firing threshold:

.. math::
    C_{mem} \cdot \frac{dU}{dt} = -G_{mem}\cdot U + \sum I_{syn} + I_{bias} + I_{app}

    \tau_{\theta}\frac{d\theta}{dt} = -\theta + \theta_0 + m\cdot U

where :math:`\tau_{\theta}` is a threshold time constant, :math:`\theta_0` is an initial threshold voltage, and :math:`m`
is a proportionality constant describing how changes in :math:`U` affect :math:`\theta`. These dynamics produce spiking
behavior using the spiking variable :math:`\delta`, which represents a spike and resets the membrane state:

.. math::
    \delta =
    \begin{cases}
        1, & U\geq\theta\\
        0, & \text{otherwise}.
    \end{cases}

    \text{if \delta=1, 0\leftarrow U.}

Default values are as follows:

- :math:`\tau_{\theta} = 5 ms`
- :math:`\theta_0 = 1 mV`
- :math:`m = 0`

This neuron can be implemented using
`sns_toolbox.design.neurons.SpikingNeuron <https://sns-toolbox.readthedocs.io/en/latest/autoapi/sns_toolbox/design/neurons/index.html#sns_toolbox.design.neurons.SpikingNeuron>`_.

===========================================
Non-Spiking Neuron with Gated Ion Channels:
===========================================

These neurons share the same dynamics as the classic non-spiking neuron, with additional ionic currents :math:`I_{ion}`.

.. math::
    C_{mem} \cdot \frac{dU}{dt} = -G_{mem}\cdot U + \sum I_{syn} + I_{bias} + I_{app} + I_{ion}

:math:`I_{ion}` represents currents flowing through voltage-gated ion channels, which can be responsible for additional
nonlinear behavior:

.. math::
    I_{ion} = \sum_j G_{ion,j} \cdot a_{\infty,j}(U) \cdot b_j^{p_{b,j}}  \cdot c_j^{p_{c,j}} \cdot \left ( \Delta E_{ion,j}-U \right )

Any neuron within a network can have any number of ion channels. :math:`G_{ion,j}` is the maximum ionic conductance of
the j\ :sup:`th` ion channel, and :math:`\Delta E_{ion,j}` is the relative ionic reversal potential (difference between the
reversal potential and neural resting potential). :math:`b` and :math:`c` are dynamical gating variables, and have the
following dynamics:

.. math::
    \frac{dz_j}{dt} = \frac{z_{\infty,j}(U) - z_j}{\tau_{z,j}(U)},

where functions of the form :math:`z_{\infty,j}` are a voltage-dependent steady-state

.. math::
    z_{\infty,j} = \frac{1}{1 + K_{z,j} \cdot \text{exp}\left ( S_{z,j} \cdot \left ( \Delta E_{z,j} - U \right ) \right )}

and :math:`\tau_{z,j}(U)` is a voltage-dependent time constant

.. math::
    \tau_{z,j}(U) = \tau_{max,z,j} \cdot z_{\infty,j}(U) \cdot \sqrt{K_{z,j} \cdot \text{exp}\left ( S_{z,j} \cdot \left ( \Delta E_{z,j} - U \right ) \right )}.

:math:`p` denotes an exponent, and :math:`\Delta E_{z,j}` is the gate relative reversal potential. :math:`K_{z,j}` and
:math:`S_{z,j}` are parameters for shaping the :math:`z_{\infty,j}(U)` and :math:`\tau_{z,j}(U)` curves.
:math:`\tau_{max,z,j}` is the maximum value of :math:`\tau_{z,j}(U)`.

This neuron can be implemented using `sns_toolbox.design.neurons.NonSpikingNeuronWithGatedChannels <https://sns-toolbox.readthedocs.io/en/latest/autoapi/sns_toolbox/design/neurons/index.html#sns_toolbox.design.neurons.NonSpikingNeuronWithGatedChannels>`_.

===================================================
Non-Spiking Neuron with Persistent Sodium Channel:
===================================================

These neurons are a special case of the non-spiking neuron with gated ion channels:

.. math::
    C_{mem} \cdot \frac{dU}{dt} = -G_{mem}\cdot U + \sum I_{syn} + I_{bias} + I_{app} + \sum_j G_{Na,j} \cdot m_{\infty,j}(U) \cdot h_j \cdot \left ( \Delta E_{Na,j}-U \right )

Default values of the channel are as follows:

- :math:`G_{Na} = 1.049 \mu S`
- :math:`p_{h} = 1`
- :math:`\Delta E_{Na} = 110mV`
- :math:`K_m = 1`
- :math:`S_m = \frac{1}{2}`
- :math:`\Delta E_m = 20mV`
- :math:`K_h = \frac{1}{2}`
- :math:`S_h = -\frac{1}{2}`
- :math:`\Delta E_h = 0mV`
- :math:`\tau_{max,h} = 300ms`

This neuron can be implemented using `sns_toolbox.design.neurons.NonSpikingNeuronWithPersisitentSodiumChannel <https://sns-toolbox.readthedocs.io/en/latest/autoapi/sns_toolbox/design/neurons/index.html#sns_toolbox.design.neurons.NonSpikingNeuronWithPersistentSodiumChannel>`_
