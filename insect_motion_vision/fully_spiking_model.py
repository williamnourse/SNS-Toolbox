"""
Recreating the vision model from Anna Sedlackova's master's thesis
William Nourse
January 5th 2022
"""

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
IMPORTS
"""

from sns_toolbox.design.neurons import SpikingNeuron, NonSpikingNeuron
from sns_toolbox.design.connections import SpikingSynapse
from sns_toolbox.design.networks import Network

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MODEL COMPONENTS
"""
# General
max_frequency = 0.1     # kHz
voltage_range = 20.0    # mV
tau_syn = 2.17          # ms
del_e_ex = 160.0        # mV
del_e_in = -80.0        # mV
dt = 0.1

neuron_spiking_general = SpikingNeuron(membrane_capacitance=200,    # nF
                                       threshold_initial_value=1.0, # mV
                                       bias=0.5)                    # nA

# Retina to Lamina
synapse_retina_lamina_excitatory = SpikingSynapse(max_conductance=1.0635,               # uS
                                                  relative_reversal_potential=del_e_ex, # mV
                                                  time_constant=tau_syn,                # ms
                                                  R=voltage_range)                      # mV

synapse_retina_lamina_inhibitory = SpikingSynapse(max_conductance=0.3072,               # uS
                                                  relative_reversal_potential=del_e_in, # mV
                                                  time_constant=tau_syn,                # ms
                                                  R=voltage_range)                      # mV

# Lamina to Medulla
synapse_lamina_medulla_excitatory_slow = SpikingSynapse(max_conductance=0.6583,                 # uS
                                                        relative_reversal_potential=del_e_ex,   # mV
                                                        time_constant=tau_syn,                  # ms
                                                        R=voltage_range,                        # mV
                                                        transmission_delay=int(100/dt))         # dt

synapse_general_excitatory = SpikingSynapse(max_conductance=0.6583,                  # uS
                                            relative_reversal_potential=del_e_ex,    # mV
                                            time_constant=tau_syn,                   # ms
                                            R=voltage_range)                         # mV

synapse_general_inhibitory = SpikingSynapse(max_conductance=1.5361,                  # uS
                                            relative_reversal_potential=del_e_in,    # mV
                                            time_constant=tau_syn,                   # ms
                                            R=voltage_range)                         # mV

neuron_spiking_medulla = SpikingNeuron(membrane_capacitance=200,    # nF
                                       threshold_initial_value=1.0, # mV
                                       bias=-19.5)                  # nA

# Small Field to Motors
synapse_small_target = SpikingSynapse(max_conductance=0.04,
                                      relative_reversal_potential=del_e_ex,
                                      time_constant=tau_syn,
                                      R=voltage_range)

synapse_target_motor_min = 0.0
synapse_target_motor_max = 0.013

neuron_nonspiking_motor = NonSpikingNeuron(membrane_capacitance=200,
                                           bias=0.0)


"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
NETWORK CONSTRUCTION
"""

number_columns = 5
net = Network(name='Spiking Vision Network', R=voltage_range)

net.add_neuron(neuron_type=neuron_spiking_general,name='lptc_a',color='red')
net.add_neuron(neuron_type=neuron_spiking_general,name='lptc_b',color='green')

net.add_neuron(neuron_type=neuron_nonspiking_motor,name='motor_a',color='darkred')
net.add_neuron(neuron_type=neuron_nonspiking_motor,name='motor_b',color='darkgreen')

# Neurons
for i in range(number_columns):
    # Retina
    net.add_neuron(neuron_type=neuron_spiking_general,name='retina_'+str(i),color='blue')

    # Lamina
    net.add_neuron(neuron_type=neuron_spiking_general, name='lamina_' + str(i), color='purple')

    # Medulla
    net.add_neuron(neuron_type=neuron_spiking_medulla,name='medulla_a_' + str(i), color='lightsalmon')
    net.add_neuron(neuron_type=neuron_spiking_medulla, name='medulla_b_' + str(i), color='lightgreen')

    # Small Field
    net.add_neuron(neuron_type=neuron_spiking_general, name='small_field_a_' + str(i), color='pink')
    net.add_neuron(neuron_type=neuron_spiking_general, name='small_field_b_' + str(i), color='seagreen')

    # Targets
    net.add_neuron(neuron_type=neuron_spiking_medulla, name='target_a_' + str(i), color='palevioletred')
    net.add_neuron(neuron_type=neuron_spiking_medulla, name='target_b_' + str(i), color='palegreen')

# Retina to Lamina Connections
for i in range(number_columns):
    net.add_synapse(synapse_retina_lamina_excitatory,source='retina_'+str(i),destination='lamina_'+str(i),
                    name='r_l_ex',view_label=True)
    if i > 0:
        net.add_synapse(synapse_retina_lamina_inhibitory,source='retina_'+str(i-1),destination='lamina_'+str(i),
                        name='r_l_in', view_label=True)
    if i < (number_columns-1):
        net.add_synapse(synapse_retina_lamina_inhibitory, source='retina_' + str(i + 1), destination='lamina_' + str(i),
                        name='r_l_in', view_label=True)

# Lamina to Medulla Connections
for i in range(number_columns):
    net.add_synapse(synapse_lamina_medulla_excitatory_slow,source='lamina_'+str(i),destination='medulla_a_'+str(i),
                    name='ex_slow',view_label=True)
    net.add_synapse(synapse_general_excitatory, source='lamina_' + str(i),destination='medulla_b_' + str(i),
                    name='ex',view_label=True)
    if i > 0:
        net.add_synapse(synapse_general_inhibitory,source='lamina_'+str(i-1),destination='medulla_a_'+str(i),
                        name='in',view_label=True)
    if i < (number_columns-1):
        net.add_synapse(synapse_general_excitatory,source='lamina_'+str(i+1),destination='medulla_a_'+str(i),
                        name='ex',view_label=True)
    if i < (number_columns-2):
        net.add_synapse(synapse_general_inhibitory,source='lamina_'+str(i+2),destination='medulla_b_'+str(i),
                        name='in',view_label=True)


net.render_graph(view=True)
