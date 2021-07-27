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

from pygenn.genn_model import GeNNModel, create_custom_neuron_class, create_custom_weight_update_class
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
                                             extra_global_params=[("Iapp", "float")],
                                             sim_code="$(U)+= (-$(Gm)*$(U) + $(Ibias) + $(Iapp) + $(Isyn))*(DT/$(Cm));",
                                             param_names=['Gm','Ibias','Cm'],
                                             var_name_types=[("U", "scalar", VarAccess_READ_WRITE)]
                                             )

"""
Let's define a static neuron model, which has no dynamics. It's just a vessel for us to inject values at runtime
"""
inputModel = create_custom_neuron_class("input",
                                        var_name_types=[('Val','scalar', VarAccess_READ_WRITE)])

"""
Non-spiking synaptic conductance model:
Gsyn = Gmax*max(0,min(1,Upre/R))
"""
piecewiseLinearUpdate = create_custom_weight_update_class("piecewiseLinear",
                                                          param_names=['Gmax','R'],
                                                          var_name_types=[('Gsyn', 'scalar')],
                                                          event_code='$(addToInSyn, $(Gsyn)*fmin(1.0,$(U_pre)/$(R))-$(inSyn))',
                                                          event_threshold_condition_code='$(U_pre) > 0'
                                                          )
piecewiseLinearPostsynaptic = 

model = GeNNModel("float", "testNonSpike")
model.dT = 1.0  # ms

params = {"Gm": 1.0,    # uS
          "Cm": 5.0,    # nF
          "Ibias": 0.0  # nA
          }

stateInit = {"U": [0.0,1.0,2.0,3.0,4.0,5.0]}
inputInit = {'Val': [1.0,5.0]}

pop = model.add_neuron_population("Population",6, nonspikingModel, params, stateInit)
inputs = model.add_neuron_population("Inputs",2,inputModel, {},inputInit)
pop.set_extra_global_param('Iapp',0.0)

# model.add_current_source('ConstantStim',
#                          'DC',
#                          pop,
#                          {'amp': "$(stim)"},{})

model.build()

"""
########################################################################################################################
MODEL SIMULATION
########################################################################################################################
"""

model.load()

u = np.empty((50, 6))     # placeholder to store the membrane potentials of each neuron over the length of the sim
ins = np.empty((50,2))
u_view = pop.vars["U"].view # alias for the voltage states (not quite sure the point of this
in_view = inputs.vars["Val"].view

while model.t < 50.0:
    model.step_time()

    pop.pull_var_from_device("U")
    if model.t == 25.0:
        in_view[:] = [5.0,1.0]
    if model.t > 5.0:
        pop.extra_global_params['Iapp'].view[:] = 10*np.sin(0.5*model.t)
    if model.t == 10.0:
        u_view[:] = [20.0, 17.0, 14.0, 11.0, 8.0, 5.0]

    u[model.timestep -1,:] = u_view[:]
    ins[model.timestep-1,:] = in_view[:]

"""
########################################################################################################################
PLOTTING
########################################################################################################################
"""

plt.figure()
plt.plot(u)
plt.xlabel('Time (ms)')
plt.ylabel('Voltage above Rest (mV)')

plt.figure()
plt.plot(ins)
plt.show()
