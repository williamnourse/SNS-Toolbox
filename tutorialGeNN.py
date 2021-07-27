"""
Let's explore using PyGeNN to simulate a network of neurons
William Nourse
July 23rd, 2021
Thanks for joining me, keep on traveling!
"""

"""
########################################################################################################################
IMPORTS
########################################################################################################################
"""

# Standard imports
import numpy as np
import matplotlib.pyplot as plt

# PyGeNN
from pygenn.genn_model import GeNNModel, init_connectivity, create_cmlf_class,\
    create_custom_sparse_connect_init_snippet_class
from pygenn.genn_wrapper import NO_DELAY

"""
########################################################################################################################
MODEL CONSTRUCTION
########################################################################################################################
"""

# Define a custom connection pattern, where each neuron is connected to the next on in a big circle
# row_build_code is called to generate each row of the synaptic connectivity matrix (connections originating from a
# single presynaptic neuron). The mod operator ensures the connection from 9 to 0 is correct.
# For better optimization, a maximum row length is also given
ring_connection = create_custom_sparse_connect_init_snippet_class(
    "ring", # pattern name
    row_build_code=
        """
        $(addSynapse, ($(id_pre) + 1) % $(num_post));
        $(endRow);
        """,
    calc_max_row_len_func=create_cmlf_class(lambda num_pre, num_post, pars: 1)()
)

model = GeNNModel("float",  # Model simulation precision
                  "tenHH"   # Model name
                  )
model.dT = 0.1  # Simulation step size in ms

# Dictionary of neural parameters
p = {"gNa": 7.15,   # Na conductance in uS
     "ENa": 50.0,   # Na reversal potential in mV
     "gK": 1.43,    # K conductance in uS
     "EK": -95.0,   # K reversal potential in mV
     "gl": 0.02672, # leak conductance in uS
     "El": -63.563, # leak reversal potential in mV
     "C": 0.143     # membrane capacitance in nF
     }

# Dictionary of initial neural variable states
ini = {"V": -60.0,      # membrane potential
       "m": 0.0529324,  # probability for Na activation
       "h": 0.3176767,  # probability for not Na blocking
       "n": 0.5961207   # probability for K activation
       }

# Dictionary of synaptic initial states
s_ini = {"g": -0.2} # synaptic conductance in uS

# Dictionary of postsynaptic response parameters
ps_p = {"tau": 1.0, # Decay time constant in ms
        "E": -80.0  # Reversal potential in mV
        }

# Spike locations in external global memory (still unsure of what exactly this means)
stim_ini = {"startSpike": [0],
            "endSpike": [1]}

# Create a population of 10 HH neurons (using the TraubMiles model)
pop1 = model.add_neuron_population("Pop1",          # population name
                                   10,              # number of neurons in the population
                                   "TraubMiles",    # neuron type
                                   p,               # parameters
                                   ini              # initial states
                                   )

# Add a population of stimuli
stim = model.add_neuron_population("Stim",              # Name
                                   1,                   # Number of neurons
                                   "SpikeSourceArray",  # Spike generator
                                   {},                  # Parameters
                                   stim_ini             # Initial states
                                   )

# Add a population of synapses which follow the ring connection pattern
model.add_synapse_population("Pop1self",    # Name
                             "SPARSE_GLOBALG",  # Type of matrix for storage
                             10,                # Axonal delay in steps
                             pop1,              # source population
                             pop1,              # target population
                             "StaticPulse",     # Weight update model
                             {},                # Weight update parameters
                             s_ini,             # weight update initial values
                             {},                # Initial values for weight update presynaptic state variables
                             {},                # initial values for weight update postsynaptic state variables
                             "ExpCond",         # Postsynaptic response model
                             ps_p,              # parameters for postsynaptic model
                             {},                # initial values for postsynaptic model
                             init_connectivity(ring_connection,{})  # initialization snippet for sparse connections
                             )

# Connect the spike generator to the population, so the stimulator is connected to the first neuron
model.add_synapse_population("StimPop1",    # Name
                             "SPARSE_GLOBALG",  # Matrix format for storage
                             NO_DELAY,          # Length of axonal delay
                             stim,              # Source population
                             pop1,              # Destination population
                             "StaticPulse",     # Weight update model
                             {},                # Weight update parameters
                             s_ini,             # weight update initial values
                             {},                # Initial values for weight update presynaptic state variables
                             {},                # initial values for weight update postsynaptic state variables
                             "ExpCond",         # Postsynaptic response model
                             ps_p,              # parameters for postsynaptic model
                             {},                # initial values for postsynaptic model
                             init_connectivity("OneToOne",{})  # initialization snippet for sparse connections
                             )

# Set firing timestep of spike generator
stim.set_extra_global_param("spikeTimes", [0.0])

"""
########################################################################################################################
MODEL SIMULATION
########################################################################################################################
"""

model.build()   # Compile the model into CUDA/C++ code
model.load()    # Load the model into a usable state

v = np.empty((2000, 10))     # placeholder to store the membrane potentials of each neuron over the length of the sim
v_view = pop1.vars["V"].view # alias for the voltage states (not quite sure the point of this

# Run the model for 200 ms
while model.t < 200.0:
    model.step_time()   # advance the model by 1 timestep

    pop1.pull_var_from_device("V")  # pull the voltage states from the model, wherever it is

    v[model.timestep-1,:] = v_view[:]   # save the voltages to memory

"""
########################################################################################################################
PLOTTING
########################################################################################################################
"""
plt.figure()
plt.plot(v)
plt.xlabel('Time (0.1 ms)')
plt.ylabel('Membrane Voltage (mV)')
plt.show()
