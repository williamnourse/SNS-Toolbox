# cython: embedsignature=True
from cpython.exc cimport PyErr_CheckSignals
from libcpp.vector cimport vector
from libcpp.map cimport map, pair
from libcpp cimport bool
from libcpp.string cimport string
from math import ceil
import numpy as np
import sys
cimport numpy as np
cimport cython

# Short names for unsigned integer types
ctypedef unsigned char _ann_uint8
ctypedef unsigned short _ann_uint16
ctypedef unsigned int _ann_uint32
ctypedef unsigned long _ann_uint64

import ANNarchy
from ANNarchy.core.cython_ext.Connector cimport LILConnectivity as LIL

cdef extern from "ANNarchy.h":

    # User-defined functions


    # User-defined constants


    # Data structures

    # Export Population 2 (pop2)
    cdef struct PopStruct2 :
        # Number of neurons
        int get_size()
        void set_size(int)
        # Maximum delay in steps
        int get_max_delay()
        void set_max_delay(int)
        void update_max_delay(int)
        # Activate/deactivate the population
        bool is_active()
        void set_active(bool)
        # Reset the population
        void reset()


        # Local attributes
        vector[double] get_local_attribute_all_double(string)
        double get_local_attribute_double(string, int)
        void set_local_attribute_all_double(string, vector[double])
        void set_local_attribute_double(string, int, double)





        # memory management
        long int size_in_bytes()
        void clear()


    # Export Projection 2
    cdef struct ProjStruct2 :
        # Flags
        bool _transmission
        bool _plasticity
        bool _update
        int _update_period
        long _update_offset

        # Connectivity
        bool init_from_lil(vector[int], vector[vector[int]], vector[vector[double]], vector[vector[int]])
        # Access connectivity
        vector[int] get_post_rank()
        vector[ vector[int] ] get_pre_ranks()
        vector[int] get_dendrite_pre_rank(int)
        int nb_synapses()
        int nb_dendrites()
        int dendrite_size(int)





        # Local Attributes
        vector[vector[double]] get_local_attribute_all_double(string)
        vector[double] get_local_attribute_row_double(string, int)
        double get_local_attribute_double(string, int, int)
        void set_local_attribute_all_double(string, vector[vector[double]])
        void set_local_attribute_row_double(string, int, vector[double])
        void set_local_attribute_double(string, int, int, double)





        # cuda configuration


        # memory management
        long int size_in_bytes()
        void clear()



    # Monitors
    cdef cppclass Monitor:
        vector[int] ranks
        int period_
        int period_offset_
        long offset_


    # Population 2 (pop2) : Monitor
    cdef cppclass PopRecorder2 (Monitor):
        @staticmethod
        int create_instance(vector[int], int, int, long)
        @staticmethod
        PopRecorder2* get_instance(int)
        long int size_in_bytes()
        void clear()

        vector[vector[double]] v
        bool record_v

        vector[vector[double]] r
        bool record_r

        # Targets
        vector[vector[double]] _sum_inh
        bool record__sum_inh

    # Projection 2 : Monitor
    cdef cppclass ProjRecorder2 (Monitor):
        @staticmethod
        int create_instance(vector[int], int, int, long)
        @staticmethod
        ProjRecorder2* get_instance(int)

        vector[vector[vector[double]]] w
        bool record_w


    # Instances

    PopStruct2 pop2

    ProjStruct2 proj2

    # Methods
    void initialize(double)
    void setSeed(long, int, bool)
    void run(int nbSteps) nogil
    int run_until(int steps, vector[int] populations, bool or_and)
    void step()

    # Time
    long getTime()
    void setTime(long)

    # dt
    double getDt()
    void setDt(double dt_)


    # Number of threads
    void setNumberThreads(int, vector[int])


# Profiling (if needed)


# Population wrappers

# Wrapper for population 2 (pop2)
@cython.auto_pickle(True)
cdef class pop2_wrapper :

    def __init__(self, size, max_delay):

        pop2.set_size(size)
        pop2.set_max_delay(max_delay)
    # Number of neurons
    property size:
        def __get__(self):
            return pop2.get_size()
    # Reset the population
    def reset(self):
        pop2.reset()
    # Set the maximum delay of outgoing projections
    def set_max_delay(self, val):
        pop2.set_max_delay(val)
    # Updates the maximum delay of outgoing projections and rebuilds the arrays
    def update_max_delay(self, val):
        pop2.update_max_delay(val)
    # Allows the population to compute
    def activate(self, bool val):
        pop2.set_active(val)


    # Local Attribute
    def get_local_attribute_all(self, name, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return np.array(pop2.get_local_attribute_all_double(cpp_string))


    def get_local_attribute(self, name, rk, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return pop2.get_local_attribute_double(cpp_string, rk)


    def set_local_attribute_all(self, name, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            pop2.set_local_attribute_all_double(cpp_string, value)


    def set_local_attribute(self, name, rk, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            pop2.set_local_attribute_double(cpp_string, rk, value)







    # memory management
    def size_in_bytes(self):
        return pop2.size_in_bytes()

    def clear(self):
        return pop2.clear()


# Projection wrappers

# Wrapper for projection 2
@cython.auto_pickle(True)
cdef class proj2_wrapper :

    def __init__(self, ):
                    pass


    def init_from_lil_connectivity(self, synapses):
        " synapses is an instance of LILConnectivity "
        return proj2.init_from_lil(synapses.post_rank, synapses.pre_rank, synapses.w, synapses.delay)

    def init_from_lil(self, post_rank, pre_rank, w, delay):
        return proj2.init_from_lil(post_rank, pre_rank, w, delay)


    property size:
        def __get__(self):
            return proj2.nb_dendrites()

    # Transmission flag
    def _get_transmission(self):
        return proj2._transmission
    def _set_transmission(self, bool l):
        proj2._transmission = l

    # Update flag
    def _get_update(self):
        return proj2._update
    def _set_update(self, bool l):
        proj2._update = l

    # Plasticity flag
    def _get_plasticity(self):
        return proj2._plasticity
    def _set_plasticity(self, bool l):
        proj2._plasticity = l

    # Update period
    def _get_update_period(self):
        return proj2._update_period
    def _set_update_period(self, int l):
        proj2._update_period = l

    # Update offset
    def _get_update_offset(self):
        return proj2._update_offset
    def _set_update_offset(self, long l):
        proj2._update_offset = l

    # Access connectivity

    def post_rank(self):
        return proj2.get_post_rank()
    def pre_rank_all(self):
        return proj2.get_pre_ranks()
    def pre_rank(self, int n):
        return proj2.get_dendrite_pre_rank(n)
    def nb_dendrites(self):
        return proj2.nb_dendrites()
    def nb_synapses(self):
        return proj2.nb_synapses()
    def dendrite_size(self, int n):
        return proj2.dendrite_size(n)




    # Local Attribute
    def get_local_attribute_all(self, name, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return proj2.get_local_attribute_all_double(cpp_string)


    def get_local_attribute_row(self, name, rk_post, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return proj2.get_local_attribute_row_double(cpp_string, rk_post)


    def get_local_attribute(self, name, rk_post, rk_pre, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return proj2.get_local_attribute_double(cpp_string, rk_post, rk_pre)


    def set_local_attribute_all(self, name, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            proj2.set_local_attribute_all_double(cpp_string, value)


    def set_local_attribute_row(self, name, rk_post, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            proj2.set_local_attribute_row_double(cpp_string, rk_post, value)


    def set_local_attribute(self, name, rk_post, rk_pre, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            proj2.set_local_attribute_double(cpp_string, rk_post, rk_pre, value)






        # cuda configuration


    # memory management
    def size_in_bytes(self):
        return proj2.size_in_bytes()

    def clear(self):
        return proj2.clear()


# Monitor wrappers

# Population Monitor wrapper
@cython.auto_pickle(True)
cdef class PopRecorder2_wrapper:
    cdef int id
    def __init__(self, list ranks, int period, period_offset, long offset):
        self.id = PopRecorder2.create_instance(ranks, period, period_offset, offset)

    def size_in_bytes(self):
        return (PopRecorder2.get_instance(self.id)).size_in_bytes()

    def clear(self):
        return (PopRecorder2.get_instance(self.id)).clear()

    property period:
        def __get__(self): return (PopRecorder2.get_instance(self.id)).period_
        def __set__(self, val): (PopRecorder2.get_instance(self.id)).period_ = val

    property period_offset:
        def __get__(self): return (PopRecorder2.get_instance(self.id)).period_offset_
        def __set__(self, val): (PopRecorder2.get_instance(self.id)).period_offset_ = val

    property v:
        def __get__(self): return (PopRecorder2.get_instance(self.id)).v
        def __set__(self, val): (PopRecorder2.get_instance(self.id)).v = val
    property record_v:
        def __get__(self): return (PopRecorder2.get_instance(self.id)).record_v
        def __set__(self, val): (PopRecorder2.get_instance(self.id)).record_v = val
    def clear_v(self):
        (PopRecorder2.get_instance(self.id)).v.clear()

    property r:
        def __get__(self): return (PopRecorder2.get_instance(self.id)).r
        def __set__(self, val): (PopRecorder2.get_instance(self.id)).r = val
    property record_r:
        def __get__(self): return (PopRecorder2.get_instance(self.id)).record_r
        def __set__(self, val): (PopRecorder2.get_instance(self.id)).record_r = val
    def clear_r(self):
        (PopRecorder2.get_instance(self.id)).r.clear()

    # Targets
    property _sum_inh:
        def __get__(self): return (PopRecorder2.get_instance(self.id))._sum_inh
        def __set__(self, val): (PopRecorder2.get_instance(self.id))._sum_inh = val
    property record__sum_inh:
        def __get__(self): return (PopRecorder2.get_instance(self.id)).record__sum_inh
        def __set__(self, val): (PopRecorder2.get_instance(self.id)).record__sum_inh = val
    def clear__sum_inh(self):
        (PopRecorder2.get_instance(self.id))._sum_inh.clear()

# Projection Monitor wrapper
@cython.auto_pickle(True)
cdef class ProjRecorder2_wrapper:
    cdef int id
    def __init__(self, list ranks, int period, int period_offset, long offset):
        self.id = ProjRecorder2.create_instance(ranks, period, period_offset, offset)

    property w:
        def __get__(self): return (ProjRecorder2.get_instance(self.id)).w
        def __set__(self, val): (ProjRecorder2.get_instance(self.id)).w = val
    property record_w:
        def __get__(self): return (ProjRecorder2.get_instance(self.id)).record_w
        def __set__(self, val): (ProjRecorder2.get_instance(self.id)).record_w = val
    def clear_w(self):
        (ProjRecorder2.get_instance(self.id)).w.clear()


# User-defined functions


# User-defined constants


# Initialize the network
def pyx_create(double dt):
    initialize(dt)

# Simple progressbar on the command line
def progress(count, total, status=''):
    """
    Prints a progress bar on the command line.

    adapted from: https://gist.github.com/vladignatyev/06860ec2040cb497f0f3

    Modification: The original code set the '\r' at the end, so the bar disappears when finished.
    I moved it to the front, so the last status remains.
    """
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('\r[%s] %s%s ...%s' % (bar, percents, '%', status))
    sys.stdout.flush()

# Simulation for the given number of steps
def pyx_run(int nb_steps, progress_bar):
    cdef int nb, rest
    cdef int batch = 1000
    if nb_steps < batch:
        with nogil:
            run(nb_steps)
    else:
        nb = int(nb_steps/batch)
        rest = nb_steps % batch
        for i in range(nb):
            with nogil:
                run(batch)
            PyErr_CheckSignals()
            if nb > 1 and progress_bar:
                progress(i+1, nb, 'simulate()')
        if rest > 0:
            run(rest)

        if (progress_bar):
            print('\n')

# Simulation for the given number of steps except if a criterion is reached
def pyx_run_until(int nb_steps, list populations, bool mode):
    cdef int nb
    nb = run_until(nb_steps, populations, mode)
    return nb

# Simulate for one step
def pyx_step():
    step()

# Access time
def set_time(t):
    setTime(t)
def get_time():
    return getTime()

# Access dt
def set_dt(double dt):
    setDt(dt)
def get_dt():
    return getDt()


# Set number of threads
def set_number_threads(int n, core_list):
    setNumberThreads(n, core_list)


# Set seed
def set_seed(long seed, int num_sources, use_seed_seq):
    setSeed(seed, num_sources, use_seed_seq)
