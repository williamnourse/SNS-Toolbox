#include <ctime>
// typedefs need to be outside the include guards to
// be visible to all files including objects.h
typedef double randomNumber_t;  // random number type

#ifndef _BRIAN_OBJECTS_H
#define _BRIAN_OBJECTS_H

#include<vector>
#include<stdint.h>
#include "synapses_classes.h"
#include "brianlib/clocks.h"
#include "network.h"
#include "rand.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <curand.h>
#include <curand_kernel.h>

namespace brian {

extern size_t used_device_memory;

//////////////// clocks ///////////////////

//////////////// networks /////////////////

//////////////// dynamic arrays 1d ///////////
extern thrust::host_vector<int32_t> _dynamic_array_synapses__synaptic_post;
extern thrust::device_vector<int32_t> dev_dynamic_array_synapses__synaptic_post;
extern thrust::host_vector<int32_t> _dynamic_array_synapses__synaptic_pre;
extern thrust::device_vector<int32_t> dev_dynamic_array_synapses__synaptic_pre;
extern thrust::host_vector<int32_t> _dynamic_array_synapses_N_incoming;
extern thrust::device_vector<int32_t> dev_dynamic_array_synapses_N_incoming;
extern thrust::host_vector<int32_t> _dynamic_array_synapses_N_outgoing;
extern thrust::device_vector<int32_t> dev_dynamic_array_synapses_N_outgoing;

//////////////// arrays ///////////////////
extern double * _array_defaultclock_dt;
extern double * dev_array_defaultclock_dt;
extern __device__ double *d_array_defaultclock_dt;
extern const int _num__array_defaultclock_dt;
extern double * _array_defaultclock_t;
extern double * dev_array_defaultclock_t;
extern __device__ double *d_array_defaultclock_t;
extern const int _num__array_defaultclock_t;
extern int64_t * _array_defaultclock_timestep;
extern int64_t * dev_array_defaultclock_timestep;
extern __device__ int64_t *d_array_defaultclock_timestep;
extern const int _num__array_defaultclock_timestep;
extern int32_t * _array_neurongroup_i;
extern int32_t * dev_array_neurongroup_i;
extern __device__ int32_t *d_array_neurongroup_i;
extern const int _num__array_neurongroup_i;
extern double * _array_neurongroup_I;
extern double * dev_array_neurongroup_I;
extern __device__ double *d_array_neurongroup_I;
extern const int _num__array_neurongroup_I;
extern double * _array_neurongroup_Ibias;
extern double * dev_array_neurongroup_Ibias;
extern __device__ double *d_array_neurongroup_Ibias;
extern const int _num__array_neurongroup_Ibias;
extern double * _array_neurongroup_Isyn;
extern double * dev_array_neurongroup_Isyn;
extern __device__ double *d_array_neurongroup_Isyn;
extern const int _num__array_neurongroup_Isyn;
extern double * _array_neurongroup_v;
extern double * dev_array_neurongroup_v;
extern __device__ double *d_array_neurongroup_v;
extern const int _num__array_neurongroup_v;
extern int32_t * _array_synapses_N;
extern int32_t * dev_array_synapses_N;
extern __device__ int32_t *d_array_synapses_N;
extern const int _num__array_synapses_N;
extern int32_t * _array_synapses_sources;
extern int32_t * dev_array_synapses_sources;
extern __device__ int32_t *d_array_synapses_sources;
extern const int _num__array_synapses_sources;
extern int32_t * _array_synapses_targets;
extern int32_t * dev_array_synapses_targets;
extern __device__ int32_t *d_array_synapses_targets;
extern const int _num__array_synapses_targets;

//////////////// eventspaces ///////////////

//////////////// dynamic arrays 2d /////////

/////////////// static arrays /////////////
extern double *_static_array__array_neurongroup_I;
extern double *dev_static_array__array_neurongroup_I;
extern __device__ double *d_static_array__array_neurongroup_I;
extern const int _num__static_array__array_neurongroup_I;
extern int32_t *_static_array__array_synapses_sources;
extern int32_t *dev_static_array__array_synapses_sources;
extern __device__ int32_t *d_static_array__array_synapses_sources;
extern const int _num__static_array__array_synapses_sources;
extern int32_t *_static_array__array_synapses_targets;
extern int32_t *dev_static_array__array_synapses_targets;
extern __device__ int32_t *d_static_array__array_synapses_targets;
extern const int _num__static_array__array_synapses_targets;

//////////////// synapses /////////////////

// Profiling information for each code object

//////////////// random numbers /////////////////
extern curandGenerator_t curand_generator;
extern unsigned long long* dev_curand_seed;
extern __device__ unsigned long long* d_curand_seed;

extern curandState* dev_curand_states;
extern __device__ curandState* d_curand_states;
extern RandomNumberBuffer random_number_buffer;

//CUDA
extern int num_parallel_blocks;
extern int max_threads_per_block;
extern int max_threads_per_sm;
extern int max_shared_mem_size;
extern int num_threads_per_warp;

}

void _init_arrays();
void _load_arrays();
void _write_arrays();
void _dealloc_arrays();

#endif


