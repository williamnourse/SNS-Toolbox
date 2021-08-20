"""
Defining custom types, we'll use PyGeNN to simulate a network of our non-spiking neurons
William Nourse
July 23rd, 2021
Put that thing back where it came from, or so help me!
"""

"""
########################################################################################################################
IMPORTS
########################################################################################################################
"""

import numpy as np
import matplotlib.pyplot as plt

from pygenn.genn_model import GeNNModel, create_custom_neuron_class, create_custom_weight_update_class, create_custom_postsynaptic_class
from pygenn.genn_wrapper import NO_DELAY
from pygenn.genn_wrapper.Models import VarAccess_READ_WRITE

"""
########################################################################################################################
MODEL CONSTRUCTION
########################################################################################################################
"""

"""
Let's define a non-spiking neuron model, which obeys the following differential equation:
    U[t] = U[t-1] + dt/Cm*(-Gm*U[t-1] + Ibias + Isyn)
"""
nonspikingModel = create_custom_neuron_class("nonspiking",  # Name
                                             sim_code="$(U)+= (-$(Gm)*$(U) + $(Ibias) + $(Isyn))*(DT/$(Cm));",
                                             param_names=['Gm','Ibias','Cm'],
                                             var_name_types=[("U", "scalar", VarAccess_READ_WRITE)]
                                             )

"""
Let's define a static neuron model, which has no dynamics. It's just a vessel for us to inject values at runtime
"""
inputModel = create_custom_neuron_class("input",
                                        var_name_types=[('Val','scalar', VarAccess_READ_WRITE)])

"""
Non-spiking synapse model:
Gsyn = Gmax*max(0,min(1,Upre/R))
Isyn = Gsyn,pre*(delE,pre-U,post)
"""
piecewiseLinearUpdate = create_custom_weight_update_class("piecewiseLinearWeight",
                                                          param_names=['Gmax','R'],
                                                          synapse_dynamics_code='$(addToInSyn, $(Gmax)*fmin(1.0,fmax(0.0,$(U_pre)/$(R)))-$(inSyn));'
                                                          )
                                                          #var_name_types=[('Gsyn', 'scalar')],
                                                          #event_code='$(addToInSyn, $(Gsyn)*fmin(1.0,$(U_pre)/$(R))-$(inSyn))',
                                                          #event_threshold_condition_code='$(U_pre) > 0'
                                                          #)

piecewiseLinearPostsynaptic = create_custom_postsynaptic_class('piecewiseLinearPostSyn',
                                                               param_names=['delE'],
                                                               apply_input_code='$(Isyn) += $(inSyn)*($(delE)-$(U));')

"""
Pure input connection synapse model:
"""
inputWeightUpdate = create_custom_weight_update_class('inputWeight',
                                                      synapse_dynamics_code='$(addToInSyn, $(U_pre)-$(inSyn));')

inputPostsynaptic = create_custom_postsynaptic_class('inputPostSyn',
                                                     apply_input_code='$(Isyn) += $(inSyn);')
"""
Parameters for all of the neurons in the simulation
"""
model = GeNNModel("float", "testNonSpike")  # Create the model which will contain all of the neurons
model.dT = 1.0  # simulation timestep (ms)

# Nonspiking Neuron parameter values
neuronParams = {"Gm": 1.0,    # uS
                "Cm": 5.0,    # nF
                "Ibias": 0.0  # nA
                }

neuronInit = {"U": 0.0}
inputInit = {'Val': 0.0}

transmitWtParams = {'Gmax': 1.0,    # uS
                    'R': 20.0}      # mV
transmitPsParams = {'delE': 100.0}  # mV

inhibitWtParams = {'Gmax': 1.0,    # uS
                   'R': 20.0}      # mV
inhibitPsParams = {'delE': -100.0}  # mV

modulateWtParams = {'Gmax': 1.0,    # uS
                    'R': 20.0}      # mV
modulatePsParams = {'delE': 0.0}    # mV

#sourceNrn = model.add_neuron_population("Source Neuron",1, nonspikingModel, neuronParams, neuronInit)
appliedCurrent = model.add_neuron_population("Applied Current",1,inputModel, {},inputInit)
#destTransmit = model.add_neuron_population('Transmit Neuron',1,nonspikingModel,neuronParams,neuronInit)
#destInhibit = model.add_neuron_population('Inhibit Neuron',1,nonspikingModel,neuronParams,neuronInit)
#destModulate = model.add_neuron_population('Modulate Neuron',1,nonspikingModel,neuronParams,neuronInit)


#transmitSynapse = model.add_synapse_population('Transmit Synapse','DENSE_GLOBALG',NO_DELAY,sourceNrn,destTransmit,
#                                               piecewiseLinearUpdate,transmitWtParams,{},{},{},
#                                               piecewiseLinearPostsynaptic,transmitPsParams,{})
#
#inhibitSynapse = model.add_synapse_population('Inhibit Synapse','DENSE_GLOBALG',NO_DELAY,sourceNrn,destInhibit,
#                                              piecewiseLinearUpdate,inhibitWtParams,{},{},{},
#                                              piecewiseLinearPostsynaptic,inhibitPsParams,{})
#
#modulateSynapse = model.add_synapse_population('Modulate Synapse','DENSE_GLOBALG',NO_DELAY,sourceNrn,destModulate,
#                                               piecewiseLinearUpdate,modulateWtParams,{},{},{},
#                                               piecewiseLinearPostsynaptic,modulatePsParams,{})
#appliedSource = model.add_synapse_population('Source Stimulation','DENSE_GLOBALG',NO_DELAY,appliedCurrent,sourceNrn,
#                                             inputWeightUpdate,{},{},{},{},inputPostsynaptic,{},{})
#appliedModulate = model.add_synapse_population('Modulate Stimulation','DENSE_GLOBALG',NO_DELAY,appliedCurrent,
#                                               destModulate,inputWeightUpdate,{},{},{},{},inputPostsynaptic,{},{})
#"""
#model.add_current_source('ConstantStim',
#                         'DC',
#                         pop,
#                         {'amp': "$(stim)"},{})

model.build()

"""
########################################################################################################################
MODEL SIMULATION
########################################################################################################################
"""

model.load()

length = 100.0
source = np.empty((length,1))   # placeholder to store the membrane potentials of each neuron over the length of the sim
transmit = np.empty((length,1))
inhibit = np.empty((length,1))
modulate = np.empty((length,1))
current = np.empty((length,1))

source_view = sourceNrn.vars["U"].view # alias for the voltage states (not quite sure the point of this
transmit_view = destTransmit.vars['U'].view
inhibit_view = destInhibit.vars['U'].view
modulate_view = destModulate.vars['U'].view
current_view = appliedCurrent.vars["Val"].view

while model.t < length:
    model.step_time()

    sourceNrn.pull_var_from_device("U")
    destTransmit.pull_var_from_device('U')
    destInhibit.pull_var_from_device('U')
    destModulate.pull_var_from_device('U')
    if model.t == 5.0:
        current_view[:] = 20.0
    #if model.t > 5.0:
    #    pop.extra_global_params['Iapp'].view[:] = 10*np.sin(0.5*model.t)
    #if model.t == 10.0:
    #    u_view[:] = [20.0, 17.0, 14.0, 11.0, 8.0, 5.0]

    source[model.timestep -1,:] = source_view[:]
    transmit[model.timestep-1,:] = transmit_view[:]
    inhibit[model.timestep-1,:] = inhibit_view[:]
    modulate[model.timestep-1,:] = modulate_view[:]
    current[model.timestep-1,:] = current_view[:]

"""
########################################################################################################################
PLOTTING
########################################################################################################################
"""

plt.figure()
plt.plot(source,label='Source')
plt.plot(transmit,label='Destination')
plt.xlabel('Time (ms)')
plt.ylabel('Voltage above Rest (mV)')
plt.title('Transmission')

plt.figure()
plt.plot(source,label='Source')
plt.plot(inhibit,label='Destination')
plt.xlabel('Time (ms)')
plt.ylabel('Voltage above Rest (mV)')
plt.title('Inhibition')

plt.figure()
plt.plot(source,label='Source')
plt.plot(modulate,label='Destination')
plt.xlabel('Time (ms)')
plt.ylabel('Voltage above Rest (mV)')
plt.title('Modulation')

plt.figure()
plt.plot(current)
plt.xlabel('Time (ms)')
plt.ylabel('Current (nA)')
plt.title('Applied Current')
plt.show()
