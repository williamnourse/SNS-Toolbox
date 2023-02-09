#include "code_objects/synapses_synapses_create_array_codeobject.h"
#include "objects.h"
#include "brianlib/common_math.h"
#include "brianlib/cuda_utils.h"
#include "brianlib/stdint_compat.h"
#include <cmath>
#include <stdint.h>
#include <ctime>
#include <stdio.h>


#include<map>


////// SUPPORT CODE ///////
namespace {
    randomNumber_t _host_rand(const int _vectorisation_idx);
    randomNumber_t _host_randn(const int _vectorisation_idx);
    int32_t _host_poisson(double _lambda, const int _vectorisation_idx);

    ///// block extra_device_helper /////

    ///// support_code_lines /////
        
    template < typename T1, typename T2 > struct _higher_type;
    template < > struct _higher_type<int,int> { typedef int type; };
    template < > struct _higher_type<int,long> { typedef long type; };
    template < > struct _higher_type<int,long long> { typedef long long type; };
    template < > struct _higher_type<int,float> { typedef float type; };
    template < > struct _higher_type<int,double> { typedef double type; };
    template < > struct _higher_type<long,int> { typedef long type; };
    template < > struct _higher_type<long,long> { typedef long type; };
    template < > struct _higher_type<long,long long> { typedef long long type; };
    template < > struct _higher_type<long,float> { typedef float type; };
    template < > struct _higher_type<long,double> { typedef double type; };
    template < > struct _higher_type<long long,int> { typedef long long type; };
    template < > struct _higher_type<long long,long> { typedef long long type; };
    template < > struct _higher_type<long long,long long> { typedef long long type; };
    template < > struct _higher_type<long long,float> { typedef float type; };
    template < > struct _higher_type<long long,double> { typedef double type; };
    template < > struct _higher_type<float,int> { typedef float type; };
    template < > struct _higher_type<float,long> { typedef float type; };
    template < > struct _higher_type<float,long long> { typedef float type; };
    template < > struct _higher_type<float,float> { typedef float type; };
    template < > struct _higher_type<float,double> { typedef double type; };
    template < > struct _higher_type<double,int> { typedef double type; };
    template < > struct _higher_type<double,long> { typedef double type; };
    template < > struct _higher_type<double,long long> { typedef double type; };
    template < > struct _higher_type<double,float> { typedef double type; };
    template < > struct _higher_type<double,double> { typedef double type; };
    template < typename T1, typename T2 >
    __host__ __device__ static inline typename _higher_type<T1,T2>::type
    _brian_mod(T1 x, T2 y)
    {{
        return x-y*floor(1.0*x/y);
    }}
    template < typename T1, typename T2 >
    __host__ __device__ static inline typename _higher_type<T1,T2>::type
    _brian_floordiv(T1 x, T2 y)
    {{
        return floor(1.0*x/y);
    }}
    #ifdef _MSC_VER
    #define _brian_pow(x, y) (pow((double)(x), (y)))
    #else
    #define _brian_pow(x, y) (pow((x), (y)))
    #endif
                    inline __device__ int _brian_atomicAdd(int* address, int val)
                    {
            // hardware implementation
            return atomicAdd(address, val);
                    }
                    inline __device__ float _brian_atomicAdd(float* address, float val)
                    {
            // hardware implementation
            return atomicAdd(address, val);
                    }
                    inline __device__ double _brian_atomicAdd(double* address, double val)
                    {
                            #if (__CUDA_ARCH__ >= 600)
            // hardware implementation
            return atomicAdd(address, val);
                            #else
            // software implementation
            unsigned long long int* address_as_int = (unsigned long long int*)address;
            unsigned long long int old = *address_as_int, assumed;
            do {
                assumed = old;
                old = atomicCAS(address_as_int, assumed,
                                __double_as_longlong(val +
                                       __longlong_as_double(assumed)));
            // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
            } while (assumed != old);
            return __longlong_as_double(old);
                            #endif
                    }
                    inline __device__ int _brian_atomicMul(int* address, int val)
                    {
                        // software implementation
                        int old = *address, assumed;
                        do {
                            assumed = old;
                            old = atomicCAS(address, assumed, val * assumed);
                        } while (assumed != old);
                        return old;
                    }
                    inline __device__ float _brian_atomicMul(float* address, float val)
                    {
            // software implementation
            int* address_as_int = (int*)address;
            int old = *address_as_int, assumed;
            do {
                assumed = old;
                old = atomicCAS(address_as_int, assumed,
                                __float_as_int(val *
                                       __int_as_float(assumed)));
            // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
            } while (assumed != old);
            return __int_as_float(old);
                    }
                    inline __device__ double _brian_atomicMul(double* address, double val)
                    {
            // software implementation
            unsigned long long int* address_as_int = (unsigned long long int*)address;
            unsigned long long int old = *address_as_int, assumed;
            do {
                assumed = old;
                old = atomicCAS(address_as_int, assumed,
                                __double_as_longlong(val *
                                       __longlong_as_double(assumed)));
            // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
            } while (assumed != old);
            return __longlong_as_double(old);
                    }
                    inline __device__ int _brian_atomicDiv(int* address, int val)
                    {
                        // software implementation
                        int old = *address, assumed;
                        do {
                            assumed = old;
                            old = atomicCAS(address, assumed, val / assumed);
                        } while (assumed != old);
                        return old;
                    }
                    inline __device__ float _brian_atomicDiv(float* address, float val)
                    {
            // software implementation
            int* address_as_int = (int*)address;
            int old = *address_as_int, assumed;
            do {
                assumed = old;
                old = atomicCAS(address_as_int, assumed,
                                __float_as_int(val /
                                       __int_as_float(assumed)));
            // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
            } while (assumed != old);
            return __int_as_float(old);
                    }
                    inline __device__ double _brian_atomicDiv(double* address, double val)
                    {
            // software implementation
            unsigned long long int* address_as_int = (unsigned long long int*)address;
            unsigned long long int old = *address_as_int, assumed;
            do {
                assumed = old;
                old = atomicCAS(address_as_int, assumed,
                                __double_as_longlong(val /
                                       __longlong_as_double(assumed)));
            // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
            } while (assumed != old);
            return __longlong_as_double(old);
                    }


    // Implement dummy functions such that the host compiled code of binomial
    // functions works. Hacky, hacky ...
    randomNumber_t _host_rand(const int _vectorisation_idx)
    {
        printf("ERROR: Called dummy function `_host_rand` in %s:%d\n", __FILE__,
                __LINE__);
        exit(EXIT_FAILURE);
    }
    randomNumber_t _host_randn(const int _vectorisation_idx)
    {
        printf("ERROR: Called dummy function `_host_rand` in %s:%d\n", __FILE__,
                __LINE__);
        exit(EXIT_FAILURE);
    }
    int32_t _host_poisson(double _lambda, const int _vectorisation_idx)
    {
        printf("ERROR: Called dummy function `_host_poisson` in %s:%d\n", __FILE__,
                __LINE__);
        exit(EXIT_FAILURE);
    }
}

////// hashdefine_lines ///////





void _run_synapses_synapses_create_array_codeobject()
{
    using namespace brian;

std::clock_t start_timer = std::clock();

CUDA_CHECK_MEMORY();
size_t used_device_memory_start = used_device_memory;


    ///// HOST_CONSTANTS ///////////
    const int _numN = 1;
		int32_t* const _array_synapses_N_incoming = thrust::raw_pointer_cast(&_dynamic_array_synapses_N_incoming[0]);
		const int _numN_incoming = _dynamic_array_synapses_N_incoming.size();
		int32_t* const _array_synapses_N_outgoing = thrust::raw_pointer_cast(&_dynamic_array_synapses_N_outgoing[0]);
		const int _numN_outgoing = _dynamic_array_synapses_N_outgoing.size();
		int32_t* const _array_synapses__synaptic_post = thrust::raw_pointer_cast(&_dynamic_array_synapses__synaptic_post[0]);
		const int _num_postsynaptic_idx = _dynamic_array_synapses__synaptic_post.size();
		int32_t* const _array_synapses__synaptic_pre = thrust::raw_pointer_cast(&_dynamic_array_synapses__synaptic_pre[0]);
		const int _num_presynaptic_idx = _dynamic_array_synapses__synaptic_pre.size();
		const int _num_synaptic_post = _dynamic_array_synapses__synaptic_post.size();
		const int _num_synaptic_pre = _dynamic_array_synapses__synaptic_pre.size();
		const int _numsources = 100;
		const int _numtargets = 100;



const int _N_pre = 10;
const int _N_post = 10;
_dynamic_array_synapses_N_incoming.resize(_N_post + 0);
_dynamic_array_synapses_N_outgoing.resize(_N_pre + 0);

///// pointers_lines /////

int32_t* __restrict  _ptr_array_synapses_N_incoming = _array_synapses_N_incoming;
int32_t* __restrict  _ptr_array_synapses__synaptic_pre = _array_synapses__synaptic_pre;
int32_t* __restrict  _ptr_array_synapses_targets = _array_synapses_targets;
int32_t*   _ptr_array_synapses_N = _array_synapses_N;
int32_t* __restrict  _ptr_array_synapses_sources = _array_synapses_sources;
int32_t* __restrict  _ptr_array_synapses_N_outgoing = _array_synapses_N_outgoing;
int32_t* __restrict  _ptr_array_synapses__synaptic_post = _array_synapses__synaptic_post;


for (int _idx=0; _idx<_numsources; _idx++) {
        
    const int32_t sources = _ptr_array_synapses_sources[_idx];
    const int32_t targets = _ptr_array_synapses_targets[_idx];
    const int32_t _real_sources = sources;
    const int32_t _real_targets = targets;


    _dynamic_array_synapses__synaptic_pre.push_back(_real_sources);
    _dynamic_array_synapses__synaptic_post.push_back(_real_targets);
    _dynamic_array_synapses_N_outgoing[_real_sources]++;
    _dynamic_array_synapses_N_incoming[_real_targets]++;
}

// now we need to resize all registered variables
const int32_t newsize = _dynamic_array_synapses__synaptic_pre.size();
        THRUST_CHECK_ERROR(
                dev_dynamic_array_synapses__synaptic_post.resize(newsize)
                );
        _dynamic_array_synapses__synaptic_post.resize(newsize);
        THRUST_CHECK_ERROR(
                dev_dynamic_array_synapses__synaptic_pre.resize(newsize)
                );
        _dynamic_array_synapses__synaptic_pre.resize(newsize);
CUDA_CHECK_MEMORY();

// update the total number of synapses
_ptr_array_synapses_N[0] = newsize;

// Check for occurrence of multiple source-target pairs in synapses ("synapse number")
std::map<std::pair<int32_t, int32_t>, int32_t> source_target_count;
for (int _i=0; _i<newsize; _i++)
{
    // Note that source_target_count will create a new entry initialized
    // with 0 when the key does not exist yet
    const std::pair<int32_t, int32_t> source_target = std::pair<int32_t, int32_t>(_dynamic_array_synapses__synaptic_pre[_i], _dynamic_array_synapses__synaptic_post[_i]);
    source_target_count[source_target]++;
    //printf("source target count = %i\n", source_target_count[source_target]);
    if (source_target_count[source_target] > 1)
    {
        synapses_multiple_pre_post = true;
        break;
    }
}
// Check
// copy changed host data to device
dev_dynamic_array_synapses_N_incoming = _dynamic_array_synapses_N_incoming;
dev_dynamic_array_synapses_N_outgoing = _dynamic_array_synapses_N_outgoing;
dev_dynamic_array_synapses__synaptic_pre = _dynamic_array_synapses__synaptic_pre;
dev_dynamic_array_synapses__synaptic_post = _dynamic_array_synapses__synaptic_post;
CUDA_SAFE_CALL(
        cudaMemcpy(dev_array_synapses_N,
            _array_synapses_N,
            sizeof(int32_t),
            cudaMemcpyHostToDevice)
        );






CUDA_CHECK_MEMORY();
const double to_MB = 1.0 / (1024.0 * 1024.0);
double tot_memory_MB = (used_device_memory - used_device_memory_start) * to_MB;
double time_passed = (double)(std::clock() - start_timer) / CLOCKS_PER_SEC;
std::cout << "INFO: synapses creation took " <<  time_passed << "s";
if (tot_memory_MB > 0)
    std::cout << " and used " << tot_memory_MB << "MB of device memory.";
std::cout << std::endl;
}


