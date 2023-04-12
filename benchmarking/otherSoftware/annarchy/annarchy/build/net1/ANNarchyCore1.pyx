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

    # Export Population 0 (pop0)
    cdef struct PopStruct0 :
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



        # Compute firing rate
        void compute_firing_rate(double window)


        # memory management
        long int size_in_bytes()
        void clear()

    # Export Population 1 (pop1)
    cdef struct PopStruct1 :
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



        # Compute firing rate
        void compute_firing_rate(double window)


        # memory management
        long int size_in_bytes()
        void clear()

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



        # Compute firing rate
        void compute_firing_rate(double window)


        # memory management
        long int size_in_bytes()
        void clear()

    # Export Population 3 (pop3)
    cdef struct PopStruct3 :
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



        # Compute firing rate
        void compute_firing_rate(double window)


        # memory management
        long int size_in_bytes()
        void clear()


    # Export Projection 0
    cdef struct ProjStruct0 :
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

        map[int, int] nb_efferent_synapses()





        # Local Attributes
        vector[vector[double]] get_local_attribute_all_double(string)
        vector[double] get_local_attribute_row_double(string, int)
        double get_local_attribute_double(string, int, int)
        void set_local_attribute_all_double(string, vector[vector[double]])
        void set_local_attribute_row_double(string, int, vector[double])
        void set_local_attribute_double(string, int, int, double)





        # cuda configuration
        void update_launch_config(int, int)

        # memory management
        long int size_in_bytes()
        void clear()



    # Monitors
    cdef cppclass Monitor:
        vector[int] ranks
        int period_
        int period_offset_
        long offset_


    # Population 0 (pop0) : Monitor
    cdef cppclass PopRecorder0 (Monitor):
        @staticmethod
        int create_instance(vector[int], int, int, long)
        @staticmethod
        PopRecorder0* get_instance(int)
        long int size_in_bytes()
        void clear()

        vector[vector[double]] v
        bool record_v

        vector[vector[double]] T
        bool record_T

        vector[vector[double]] g_inh
        bool record_g_inh

        vector[vector[double]] r
        bool record_r

        map[int, vector[long]] spike
        bool record_spike
        void clear_spike()

    # Population 1 (pop1) : Monitor
    cdef cppclass PopRecorder1 (Monitor):
        @staticmethod
        int create_instance(vector[int], int, int, long)
        @staticmethod
        PopRecorder1* get_instance(int)
        long int size_in_bytes()
        void clear()

        vector[vector[double]] v
        bool record_v

        vector[vector[double]] T
        bool record_T

        vector[vector[double]] g_inh
        bool record_g_inh

        vector[vector[double]] r
        bool record_r

        map[int, vector[long]] spike
        bool record_spike
        void clear_spike()

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

        vector[vector[double]] T
        bool record_T

        vector[vector[double]] g_inh
        bool record_g_inh

        vector[vector[double]] r
        bool record_r

        map[int, vector[long]] spike
        bool record_spike
        void clear_spike()

    # Population 3 (pop3) : Monitor
    cdef cppclass PopRecorder3 (Monitor):
        @staticmethod
        int create_instance(vector[int], int, int, long)
        @staticmethod
        PopRecorder3* get_instance(int)
        long int size_in_bytes()
        void clear()

        vector[vector[double]] v
        bool record_v

        vector[vector[double]] T
        bool record_T

        vector[vector[double]] g_inh
        bool record_g_inh

        vector[vector[double]] r
        bool record_r

        map[int, vector[long]] spike
        bool record_spike
        void clear_spike()

    # Projection 0 : Monitor
    cdef cppclass ProjRecorder0 (Monitor):
        @staticmethod
        int create_instance(vector[int], int, int, long)
        @staticmethod
        ProjRecorder0* get_instance(int)


    # Instances

    PopStruct0 pop0
    PopStruct1 pop1
    PopStruct2 pop2
    PopStruct3 pop3

    ProjStruct0 proj0

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


    # GPU device
    void setDevice(int)


# Profiling (if needed)


# Population wrappers

# Wrapper for population 0 (pop0)
@cython.auto_pickle(True)
cdef class pop0_wrapper :

    def __init__(self, size, max_delay):

        pop0.set_size(size)
        pop0.set_max_delay(max_delay)
    # Number of neurons
    property size:
        def __get__(self):
            return pop0.get_size()
    # Reset the population
    def reset(self):
        pop0.reset()
    # Set the maximum delay of outgoing projections
    def set_max_delay(self, val):
        pop0.set_max_delay(val)
    # Updates the maximum delay of outgoing projections and rebuilds the arrays
    def update_max_delay(self, val):
        pop0.update_max_delay(val)
    # Allows the population to compute
    def activate(self, bool val):
        pop0.set_active(val)


    # Local Attribute
    def get_local_attribute_all(self, name, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return np.array(pop0.get_local_attribute_all_double(cpp_string))


    def get_local_attribute(self, name, rk, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return pop0.get_local_attribute_double(cpp_string, rk)


    def set_local_attribute_all(self, name, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            pop0.set_local_attribute_all_double(cpp_string, value)


    def set_local_attribute(self, name, rk, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            pop0.set_local_attribute_double(cpp_string, rk, value)





    # Compute firing rate
    cpdef compute_firing_rate(self, double window):
        pop0.compute_firing_rate(window)


    # memory management
    def size_in_bytes(self):
        return pop0.size_in_bytes()

    def clear(self):
        return pop0.clear()

# Wrapper for population 1 (pop1)
@cython.auto_pickle(True)
cdef class pop1_wrapper :

    def __init__(self, size, max_delay):

        pop1.set_size(size)
        pop1.set_max_delay(max_delay)
    # Number of neurons
    property size:
        def __get__(self):
            return pop1.get_size()
    # Reset the population
    def reset(self):
        pop1.reset()
    # Set the maximum delay of outgoing projections
    def set_max_delay(self, val):
        pop1.set_max_delay(val)
    # Updates the maximum delay of outgoing projections and rebuilds the arrays
    def update_max_delay(self, val):
        pop1.update_max_delay(val)
    # Allows the population to compute
    def activate(self, bool val):
        pop1.set_active(val)


    # Local Attribute
    def get_local_attribute_all(self, name, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return np.array(pop1.get_local_attribute_all_double(cpp_string))


    def get_local_attribute(self, name, rk, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return pop1.get_local_attribute_double(cpp_string, rk)


    def set_local_attribute_all(self, name, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            pop1.set_local_attribute_all_double(cpp_string, value)


    def set_local_attribute(self, name, rk, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            pop1.set_local_attribute_double(cpp_string, rk, value)





    # Compute firing rate
    cpdef compute_firing_rate(self, double window):
        pop1.compute_firing_rate(window)


    # memory management
    def size_in_bytes(self):
        return pop1.size_in_bytes()

    def clear(self):
        return pop1.clear()

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





    # Compute firing rate
    cpdef compute_firing_rate(self, double window):
        pop2.compute_firing_rate(window)


    # memory management
    def size_in_bytes(self):
        return pop2.size_in_bytes()

    def clear(self):
        return pop2.clear()

# Wrapper for population 3 (pop3)
@cython.auto_pickle(True)
cdef class pop3_wrapper :

    def __init__(self, size, max_delay):

        pop3.set_size(size)
        pop3.set_max_delay(max_delay)
    # Number of neurons
    property size:
        def __get__(self):
            return pop3.get_size()
    # Reset the population
    def reset(self):
        pop3.reset()
    # Set the maximum delay of outgoing projections
    def set_max_delay(self, val):
        pop3.set_max_delay(val)
    # Updates the maximum delay of outgoing projections and rebuilds the arrays
    def update_max_delay(self, val):
        pop3.update_max_delay(val)
    # Allows the population to compute
    def activate(self, bool val):
        pop3.set_active(val)


    # Local Attribute
    def get_local_attribute_all(self, name, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return np.array(pop3.get_local_attribute_all_double(cpp_string))


    def get_local_attribute(self, name, rk, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return pop3.get_local_attribute_double(cpp_string, rk)


    def set_local_attribute_all(self, name, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            pop3.set_local_attribute_all_double(cpp_string, value)


    def set_local_attribute(self, name, rk, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            pop3.set_local_attribute_double(cpp_string, rk, value)





    # Compute firing rate
    cpdef compute_firing_rate(self, double window):
        pop3.compute_firing_rate(window)


    # memory management
    def size_in_bytes(self):
        return pop3.size_in_bytes()

    def clear(self):
        return pop3.clear()


# Projection wrappers

# Wrapper for projection 0
@cython.auto_pickle(True)
cdef class proj0_wrapper :

    def __init__(self, ):
                    pass


    def init_from_lil_connectivity(self, synapses):
        " synapses is an instance of LILConnectivity "
        return proj0.init_from_lil(synapses.post_rank, synapses.pre_rank, synapses.w, synapses.delay)

    def init_from_lil(self, post_rank, pre_rank, w, delay):
        return proj0.init_from_lil(post_rank, pre_rank, w, delay)


    property size:
        def __get__(self):
            return proj0.nb_dendrites()

    # Transmission flag
    def _get_transmission(self):
        return proj0._transmission
    def _set_transmission(self, bool l):
        proj0._transmission = l

    # Update flag
    def _get_update(self):
        return proj0._update
    def _set_update(self, bool l):
        proj0._update = l

    # Plasticity flag
    def _get_plasticity(self):
        return proj0._plasticity
    def _set_plasticity(self, bool l):
        proj0._plasticity = l

    # Update period
    def _get_update_period(self):
        return proj0._update_period
    def _set_update_period(self, int l):
        proj0._update_period = l

    # Update offset
    def _get_update_offset(self):
        return proj0._update_offset
    def _set_update_offset(self, long l):
        proj0._update_offset = l

    # Access connectivity

    def post_rank(self):
        return proj0.get_post_rank()
    def pre_rank_all(self):
        return proj0.get_pre_ranks()
    def pre_rank(self, int n):
        return proj0.get_dendrite_pre_rank(n)
    def nb_dendrites(self):
        return proj0.nb_dendrites()
    def nb_synapses(self):
        return proj0.nb_synapses()
    def dendrite_size(self, int n):
        return proj0.dendrite_size(n)

    def nb_efferent_synapses(self):
        return proj0.nb_efferent_synapses()




    # Local Attribute
    def get_local_attribute_all(self, name, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return proj0.get_local_attribute_all_double(cpp_string)


    def get_local_attribute_row(self, name, rk_post, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return proj0.get_local_attribute_row_double(cpp_string, rk_post)


    def get_local_attribute(self, name, rk_post, rk_pre, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return proj0.get_local_attribute_double(cpp_string, rk_post, rk_pre)


    def set_local_attribute_all(self, name, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            proj0.set_local_attribute_all_double(cpp_string, value)


    def set_local_attribute_row(self, name, rk_post, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            proj0.set_local_attribute_row_double(cpp_string, rk_post, value)


    def set_local_attribute(self, name, rk_post, rk_pre, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            proj0.set_local_attribute_double(cpp_string, rk_post, rk_pre, value)






        # cuda configuration

    def update_launch_config(self, nb_blocks=-1, threads_per_block=32):
        proj0.update_launch_config(nb_blocks, threads_per_block)


    # memory management
    def size_in_bytes(self):
        return proj0.size_in_bytes()

    def clear(self):
        return proj0.clear()


# Monitor wrappers

# Population Monitor wrapper
@cython.auto_pickle(True)
cdef class PopRecorder0_wrapper:
    cdef int id
    def __init__(self, list ranks, int period, period_offset, long offset):
        self.id = PopRecorder0.create_instance(ranks, period, period_offset, offset)

    def size_in_bytes(self):
        return (PopRecorder0.get_instance(self.id)).size_in_bytes()

    def clear(self):
        return (PopRecorder0.get_instance(self.id)).clear()

    property period:
        def __get__(self): return (PopRecorder0.get_instance(self.id)).period_
        def __set__(self, val): (PopRecorder0.get_instance(self.id)).period_ = val

    property period_offset:
        def __get__(self): return (PopRecorder0.get_instance(self.id)).period_offset_
        def __set__(self, val): (PopRecorder0.get_instance(self.id)).period_offset_ = val

    property v:
        def __get__(self): return (PopRecorder0.get_instance(self.id)).v
        def __set__(self, val): (PopRecorder0.get_instance(self.id)).v = val
    property record_v:
        def __get__(self): return (PopRecorder0.get_instance(self.id)).record_v
        def __set__(self, val): (PopRecorder0.get_instance(self.id)).record_v = val
    def clear_v(self):
        (PopRecorder0.get_instance(self.id)).v.clear()

    property T:
        def __get__(self): return (PopRecorder0.get_instance(self.id)).T
        def __set__(self, val): (PopRecorder0.get_instance(self.id)).T = val
    property record_T:
        def __get__(self): return (PopRecorder0.get_instance(self.id)).record_T
        def __set__(self, val): (PopRecorder0.get_instance(self.id)).record_T = val
    def clear_T(self):
        (PopRecorder0.get_instance(self.id)).T.clear()

    property g_inh:
        def __get__(self): return (PopRecorder0.get_instance(self.id)).g_inh
        def __set__(self, val): (PopRecorder0.get_instance(self.id)).g_inh = val
    property record_g_inh:
        def __get__(self): return (PopRecorder0.get_instance(self.id)).record_g_inh
        def __set__(self, val): (PopRecorder0.get_instance(self.id)).record_g_inh = val
    def clear_g_inh(self):
        (PopRecorder0.get_instance(self.id)).g_inh.clear()

    property r:
        def __get__(self): return (PopRecorder0.get_instance(self.id)).r
        def __set__(self, val): (PopRecorder0.get_instance(self.id)).r = val
    property record_r:
        def __get__(self): return (PopRecorder0.get_instance(self.id)).record_r
        def __set__(self, val): (PopRecorder0.get_instance(self.id)).record_r = val
    def clear_r(self):
        (PopRecorder0.get_instance(self.id)).r.clear()

    property spike:
        def __get__(self): return (PopRecorder0.get_instance(self.id)).spike
        def __set__(self, val): (PopRecorder0.get_instance(self.id)).spike = val
    property record_spike:
        def __get__(self): return (PopRecorder0.get_instance(self.id)).record_spike
        def __set__(self, val): (PopRecorder0.get_instance(self.id)).record_spike = val
    def clear_spike(self):
        (PopRecorder0.get_instance(self.id)).clear_spike()

# Population Monitor wrapper
@cython.auto_pickle(True)
cdef class PopRecorder1_wrapper:
    cdef int id
    def __init__(self, list ranks, int period, period_offset, long offset):
        self.id = PopRecorder1.create_instance(ranks, period, period_offset, offset)

    def size_in_bytes(self):
        return (PopRecorder1.get_instance(self.id)).size_in_bytes()

    def clear(self):
        return (PopRecorder1.get_instance(self.id)).clear()

    property period:
        def __get__(self): return (PopRecorder1.get_instance(self.id)).period_
        def __set__(self, val): (PopRecorder1.get_instance(self.id)).period_ = val

    property period_offset:
        def __get__(self): return (PopRecorder1.get_instance(self.id)).period_offset_
        def __set__(self, val): (PopRecorder1.get_instance(self.id)).period_offset_ = val

    property v:
        def __get__(self): return (PopRecorder1.get_instance(self.id)).v
        def __set__(self, val): (PopRecorder1.get_instance(self.id)).v = val
    property record_v:
        def __get__(self): return (PopRecorder1.get_instance(self.id)).record_v
        def __set__(self, val): (PopRecorder1.get_instance(self.id)).record_v = val
    def clear_v(self):
        (PopRecorder1.get_instance(self.id)).v.clear()

    property T:
        def __get__(self): return (PopRecorder1.get_instance(self.id)).T
        def __set__(self, val): (PopRecorder1.get_instance(self.id)).T = val
    property record_T:
        def __get__(self): return (PopRecorder1.get_instance(self.id)).record_T
        def __set__(self, val): (PopRecorder1.get_instance(self.id)).record_T = val
    def clear_T(self):
        (PopRecorder1.get_instance(self.id)).T.clear()

    property g_inh:
        def __get__(self): return (PopRecorder1.get_instance(self.id)).g_inh
        def __set__(self, val): (PopRecorder1.get_instance(self.id)).g_inh = val
    property record_g_inh:
        def __get__(self): return (PopRecorder1.get_instance(self.id)).record_g_inh
        def __set__(self, val): (PopRecorder1.get_instance(self.id)).record_g_inh = val
    def clear_g_inh(self):
        (PopRecorder1.get_instance(self.id)).g_inh.clear()

    property r:
        def __get__(self): return (PopRecorder1.get_instance(self.id)).r
        def __set__(self, val): (PopRecorder1.get_instance(self.id)).r = val
    property record_r:
        def __get__(self): return (PopRecorder1.get_instance(self.id)).record_r
        def __set__(self, val): (PopRecorder1.get_instance(self.id)).record_r = val
    def clear_r(self):
        (PopRecorder1.get_instance(self.id)).r.clear()

    property spike:
        def __get__(self): return (PopRecorder1.get_instance(self.id)).spike
        def __set__(self, val): (PopRecorder1.get_instance(self.id)).spike = val
    property record_spike:
        def __get__(self): return (PopRecorder1.get_instance(self.id)).record_spike
        def __set__(self, val): (PopRecorder1.get_instance(self.id)).record_spike = val
    def clear_spike(self):
        (PopRecorder1.get_instance(self.id)).clear_spike()

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

    property T:
        def __get__(self): return (PopRecorder2.get_instance(self.id)).T
        def __set__(self, val): (PopRecorder2.get_instance(self.id)).T = val
    property record_T:
        def __get__(self): return (PopRecorder2.get_instance(self.id)).record_T
        def __set__(self, val): (PopRecorder2.get_instance(self.id)).record_T = val
    def clear_T(self):
        (PopRecorder2.get_instance(self.id)).T.clear()

    property g_inh:
        def __get__(self): return (PopRecorder2.get_instance(self.id)).g_inh
        def __set__(self, val): (PopRecorder2.get_instance(self.id)).g_inh = val
    property record_g_inh:
        def __get__(self): return (PopRecorder2.get_instance(self.id)).record_g_inh
        def __set__(self, val): (PopRecorder2.get_instance(self.id)).record_g_inh = val
    def clear_g_inh(self):
        (PopRecorder2.get_instance(self.id)).g_inh.clear()

    property r:
        def __get__(self): return (PopRecorder2.get_instance(self.id)).r
        def __set__(self, val): (PopRecorder2.get_instance(self.id)).r = val
    property record_r:
        def __get__(self): return (PopRecorder2.get_instance(self.id)).record_r
        def __set__(self, val): (PopRecorder2.get_instance(self.id)).record_r = val
    def clear_r(self):
        (PopRecorder2.get_instance(self.id)).r.clear()

    property spike:
        def __get__(self): return (PopRecorder2.get_instance(self.id)).spike
        def __set__(self, val): (PopRecorder2.get_instance(self.id)).spike = val
    property record_spike:
        def __get__(self): return (PopRecorder2.get_instance(self.id)).record_spike
        def __set__(self, val): (PopRecorder2.get_instance(self.id)).record_spike = val
    def clear_spike(self):
        (PopRecorder2.get_instance(self.id)).clear_spike()

# Population Monitor wrapper
@cython.auto_pickle(True)
cdef class PopRecorder3_wrapper:
    cdef int id
    def __init__(self, list ranks, int period, period_offset, long offset):
        self.id = PopRecorder3.create_instance(ranks, period, period_offset, offset)

    def size_in_bytes(self):
        return (PopRecorder3.get_instance(self.id)).size_in_bytes()

    def clear(self):
        return (PopRecorder3.get_instance(self.id)).clear()

    property period:
        def __get__(self): return (PopRecorder3.get_instance(self.id)).period_
        def __set__(self, val): (PopRecorder3.get_instance(self.id)).period_ = val

    property period_offset:
        def __get__(self): return (PopRecorder3.get_instance(self.id)).period_offset_
        def __set__(self, val): (PopRecorder3.get_instance(self.id)).period_offset_ = val

    property v:
        def __get__(self): return (PopRecorder3.get_instance(self.id)).v
        def __set__(self, val): (PopRecorder3.get_instance(self.id)).v = val
    property record_v:
        def __get__(self): return (PopRecorder3.get_instance(self.id)).record_v
        def __set__(self, val): (PopRecorder3.get_instance(self.id)).record_v = val
    def clear_v(self):
        (PopRecorder3.get_instance(self.id)).v.clear()

    property T:
        def __get__(self): return (PopRecorder3.get_instance(self.id)).T
        def __set__(self, val): (PopRecorder3.get_instance(self.id)).T = val
    property record_T:
        def __get__(self): return (PopRecorder3.get_instance(self.id)).record_T
        def __set__(self, val): (PopRecorder3.get_instance(self.id)).record_T = val
    def clear_T(self):
        (PopRecorder3.get_instance(self.id)).T.clear()

    property g_inh:
        def __get__(self): return (PopRecorder3.get_instance(self.id)).g_inh
        def __set__(self, val): (PopRecorder3.get_instance(self.id)).g_inh = val
    property record_g_inh:
        def __get__(self): return (PopRecorder3.get_instance(self.id)).record_g_inh
        def __set__(self, val): (PopRecorder3.get_instance(self.id)).record_g_inh = val
    def clear_g_inh(self):
        (PopRecorder3.get_instance(self.id)).g_inh.clear()

    property r:
        def __get__(self): return (PopRecorder3.get_instance(self.id)).r
        def __set__(self, val): (PopRecorder3.get_instance(self.id)).r = val
    property record_r:
        def __get__(self): return (PopRecorder3.get_instance(self.id)).record_r
        def __set__(self, val): (PopRecorder3.get_instance(self.id)).record_r = val
    def clear_r(self):
        (PopRecorder3.get_instance(self.id)).r.clear()

    property spike:
        def __get__(self): return (PopRecorder3.get_instance(self.id)).spike
        def __set__(self, val): (PopRecorder3.get_instance(self.id)).spike = val
    property record_spike:
        def __get__(self): return (PopRecorder3.get_instance(self.id)).record_spike
        def __set__(self, val): (PopRecorder3.get_instance(self.id)).record_spike = val
    def clear_spike(self):
        (PopRecorder3.get_instance(self.id)).clear_spike()

# Projection Monitor wrapper
@cython.auto_pickle(True)
cdef class ProjRecorder0_wrapper:
    cdef int id
    def __init__(self, list ranks, int period, int period_offset, long offset):
        self.id = ProjRecorder0.create_instance(ranks, period, period_offset, offset)


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


# Set GPU device
def set_device(int device_id):
    setDevice(device_id)


# Set seed
def set_seed(long seed, int num_sources, use_seed_seq):
    setSeed(seed, num_sources, use_seed_seq)
