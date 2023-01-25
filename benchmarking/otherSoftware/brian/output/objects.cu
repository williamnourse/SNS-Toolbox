
#include "objects.h"
#include "synapses_classes.h"
#include "brianlib/clocks.h"
#include "brianlib/cuda_utils.h"
#include "network.h"
#include "rand.h"
#include <stdint.h>
#include <iostream>
#include <fstream>
#include <ctime>
#include <utility>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <curand.h>
#include <curand_kernel.h>

size_t brian::used_device_memory = 0;

//////////////// clocks ///////////////////

//////////////// networks /////////////////

//////////////// arrays ///////////////////
double * brian::_array_defaultclock_dt;
double * brian::dev_array_defaultclock_dt;
__device__ double * brian::d_array_defaultclock_dt;
const int brian::_num__array_defaultclock_dt = 1;

double * brian::_array_defaultclock_t;
double * brian::dev_array_defaultclock_t;
__device__ double * brian::d_array_defaultclock_t;
const int brian::_num__array_defaultclock_t = 1;

int64_t * brian::_array_defaultclock_timestep;
int64_t * brian::dev_array_defaultclock_timestep;
__device__ int64_t * brian::d_array_defaultclock_timestep;
const int brian::_num__array_defaultclock_timestep = 1;

int32_t * brian::_array_neurongroup_i;
int32_t * brian::dev_array_neurongroup_i;
__device__ int32_t * brian::d_array_neurongroup_i;
const int brian::_num__array_neurongroup_i = 10;

double * brian::_array_neurongroup_I;
double * brian::dev_array_neurongroup_I;
__device__ double * brian::d_array_neurongroup_I;
const int brian::_num__array_neurongroup_I = 10;

double * brian::_array_neurongroup_Ibias;
double * brian::dev_array_neurongroup_Ibias;
__device__ double * brian::d_array_neurongroup_Ibias;
const int brian::_num__array_neurongroup_Ibias = 10;

double * brian::_array_neurongroup_Isyn;
double * brian::dev_array_neurongroup_Isyn;
__device__ double * brian::d_array_neurongroup_Isyn;
const int brian::_num__array_neurongroup_Isyn = 10;

double * brian::_array_neurongroup_v;
double * brian::dev_array_neurongroup_v;
__device__ double * brian::d_array_neurongroup_v;
const int brian::_num__array_neurongroup_v = 10;

int32_t * brian::_array_synapses_N;
int32_t * brian::dev_array_synapses_N;
__device__ int32_t * brian::d_array_synapses_N;
const int brian::_num__array_synapses_N = 1;

int32_t * brian::_array_synapses_sources;
int32_t * brian::dev_array_synapses_sources;
__device__ int32_t * brian::d_array_synapses_sources;
const int brian::_num__array_synapses_sources = 100;

int32_t * brian::_array_synapses_targets;
int32_t * brian::dev_array_synapses_targets;
__device__ int32_t * brian::d_array_synapses_targets;
const int brian::_num__array_synapses_targets = 100;


//////////////// eventspaces ///////////////
// we dynamically create multiple eventspaces in no_or_const_delay_mode
// for initiating the first spikespace, we need a host pointer
// for choosing the right spikespace, we need a global index variable

//////////////// dynamic arrays 1d /////////
thrust::host_vector<int32_t> brian::_dynamic_array_synapses__synaptic_post;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses__synaptic_post;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses__synaptic_pre;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses__synaptic_pre;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_N_incoming;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_N_incoming;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_N_outgoing;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_N_outgoing;

//////////////// dynamic arrays 2d /////////

/////////////// static arrays /////////////
double * brian::_static_array__array_neurongroup_I;
double * brian::dev_static_array__array_neurongroup_I;
__device__ double * brian::d_static_array__array_neurongroup_I;
const int brian::_num__static_array__array_neurongroup_I = 10;
int32_t * brian::_static_array__array_synapses_sources;
int32_t * brian::dev_static_array__array_synapses_sources;
__device__ int32_t * brian::d_static_array__array_synapses_sources;
const int brian::_num__static_array__array_synapses_sources = 100;
int32_t * brian::_static_array__array_synapses_targets;
int32_t * brian::dev_static_array__array_synapses_targets;
__device__ int32_t * brian::d_static_array__array_synapses_targets;
const int brian::_num__static_array__array_synapses_targets = 100;

//////////////// synapses /////////////////

int brian::num_parallel_blocks;
int brian::max_threads_per_block;
int brian::max_threads_per_sm;
int brian::max_shared_mem_size;
int brian::num_threads_per_warp;


// Profiling information for each code object

//////////////random numbers//////////////////
curandGenerator_t brian::curand_generator;
__device__ unsigned long long* brian::d_curand_seed;
unsigned long long* brian::dev_curand_seed;
// dev_{co.name}_{rng_type}_allocator
//      pointer to start of generated random numbers array
//      at each generation cycle this array is refilled
// dev_{co.name}_{rng_type}
//      pointer moving through generated random number array
//      until it is regenerated at the next generation cycle
curandState* brian::dev_curand_states;
__device__ curandState* brian::d_curand_states;
RandomNumberBuffer brian::random_number_buffer;

void _init_arrays()
{
    using namespace brian;

    std::clock_t start_timer = std::clock();

    CUDA_CHECK_MEMORY();
    size_t used_device_memory_start = used_device_memory;

    cudaDeviceProp props;
    CUDA_SAFE_CALL(
            cudaGetDeviceProperties(&props, 0)
            );

    num_parallel_blocks = 1;
    max_threads_per_block = props.maxThreadsPerBlock;
    max_threads_per_sm = props.maxThreadsPerMultiProcessor;
    max_shared_mem_size = props.sharedMemPerBlock;
    num_threads_per_warp = props.warpSize;

    // Random seeds might be overwritten in main.cu
    unsigned long long seed = time(0);

    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_curand_seed,
                sizeof(unsigned long long))
            );

    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_curand_seed, &dev_curand_seed,
                sizeof(unsigned long long*))
            );

    CUDA_SAFE_CALL(
            curandCreateGenerator(&curand_generator, CURAND_RNG_PSEUDO_DEFAULT)
            );


    // this sets seed for host and device api RNG
    random_number_buffer.set_seed(seed);


    // Arrays initialized to 0
            _array_defaultclock_dt = new double[1];
            for(int i=0; i<1; i++) _array_defaultclock_dt[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_defaultclock_dt, sizeof(double)*_num__array_defaultclock_dt)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_defaultclock_dt, _array_defaultclock_dt, sizeof(double)*_num__array_defaultclock_dt, cudaMemcpyHostToDevice)
                    );
            _array_defaultclock_t = new double[1];
            for(int i=0; i<1; i++) _array_defaultclock_t[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_defaultclock_t, sizeof(double)*_num__array_defaultclock_t)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_defaultclock_t, _array_defaultclock_t, sizeof(double)*_num__array_defaultclock_t, cudaMemcpyHostToDevice)
                    );
            _array_defaultclock_timestep = new int64_t[1];
            for(int i=0; i<1; i++) _array_defaultclock_timestep[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_defaultclock_timestep, sizeof(int64_t)*_num__array_defaultclock_timestep)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_defaultclock_timestep, _array_defaultclock_timestep, sizeof(int64_t)*_num__array_defaultclock_timestep, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_i = new int32_t[10];
            for(int i=0; i<10; i++) _array_neurongroup_i[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_i, sizeof(int32_t)*_num__array_neurongroup_i)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_i, _array_neurongroup_i, sizeof(int32_t)*_num__array_neurongroup_i, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_I = new double[10];
            for(int i=0; i<10; i++) _array_neurongroup_I[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_I, sizeof(double)*_num__array_neurongroup_I)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_I, _array_neurongroup_I, sizeof(double)*_num__array_neurongroup_I, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_Ibias = new double[10];
            for(int i=0; i<10; i++) _array_neurongroup_Ibias[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_Ibias, sizeof(double)*_num__array_neurongroup_Ibias)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_Ibias, _array_neurongroup_Ibias, sizeof(double)*_num__array_neurongroup_Ibias, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_Isyn = new double[10];
            for(int i=0; i<10; i++) _array_neurongroup_Isyn[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_Isyn, sizeof(double)*_num__array_neurongroup_Isyn)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_Isyn, _array_neurongroup_Isyn, sizeof(double)*_num__array_neurongroup_Isyn, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_v = new double[10];
            for(int i=0; i<10; i++) _array_neurongroup_v[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_v, sizeof(double)*_num__array_neurongroup_v)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_v, _array_neurongroup_v, sizeof(double)*_num__array_neurongroup_v, cudaMemcpyHostToDevice)
                    );
            _array_synapses_N = new int32_t[1];
            for(int i=0; i<1; i++) _array_synapses_N[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_N, sizeof(int32_t)*_num__array_synapses_N)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_N, _array_synapses_N, sizeof(int32_t)*_num__array_synapses_N, cudaMemcpyHostToDevice)
                    );
            _array_synapses_sources = new int32_t[100];
            for(int i=0; i<100; i++) _array_synapses_sources[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_sources, sizeof(int32_t)*_num__array_synapses_sources)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_sources, _array_synapses_sources, sizeof(int32_t)*_num__array_synapses_sources, cudaMemcpyHostToDevice)
                    );
            _array_synapses_targets = new int32_t[100];
            for(int i=0; i<100; i++) _array_synapses_targets[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_targets, sizeof(int32_t)*_num__array_synapses_targets)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_targets, _array_synapses_targets, sizeof(int32_t)*_num__array_synapses_targets, cudaMemcpyHostToDevice)
                    );

    // Arrays initialized to an "arange"
    _array_neurongroup_i = new int32_t[10];
    for(int i=0; i<10; i++) _array_neurongroup_i[i] = 0 + i;
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_neurongroup_i, sizeof(int32_t)*_num__array_neurongroup_i)
            );

    CUDA_SAFE_CALL(
            cudaMemcpy(dev_array_neurongroup_i, _array_neurongroup_i, sizeof(int32_t)*_num__array_neurongroup_i, cudaMemcpyHostToDevice)
            );

    // static arrays
    _static_array__array_neurongroup_I = new double[10];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_static_array__array_neurongroup_I, sizeof(double)*10)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_static_array__array_neurongroup_I, &dev_static_array__array_neurongroup_I, sizeof(double*))
            );
    _static_array__array_synapses_sources = new int32_t[100];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_static_array__array_synapses_sources, sizeof(int32_t)*100)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_static_array__array_synapses_sources, &dev_static_array__array_synapses_sources, sizeof(int32_t*))
            );
    _static_array__array_synapses_targets = new int32_t[100];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_static_array__array_synapses_targets, sizeof(int32_t)*100)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_static_array__array_synapses_targets, &dev_static_array__array_synapses_targets, sizeof(int32_t*))
            );


    // eventspace_arrays

    CUDA_CHECK_MEMORY();
    const double to_MB = 1.0 / (1024.0 * 1024.0);
    double tot_memory_MB = (used_device_memory - used_device_memory_start) * to_MB;
    double time_passed = (double)(std::clock() - start_timer) / CLOCKS_PER_SEC;
    std::cout << "INFO: _init_arrays() took " <<  time_passed << "s";
    if (tot_memory_MB > 0)
        std::cout << " and used " << tot_memory_MB << "MB of device memory.";
    std::cout << std::endl;
}

void _load_arrays()
{
    using namespace brian;

    ifstream f_static_array__array_neurongroup_I;
    f_static_array__array_neurongroup_I.open("static_arrays/_static_array__array_neurongroup_I", ios::in | ios::binary);
    if(f_static_array__array_neurongroup_I.is_open())
    {
        f_static_array__array_neurongroup_I.read(reinterpret_cast<char*>(_static_array__array_neurongroup_I), 10*sizeof(double));
    } else
    {
        std::cout << "Error opening static array _static_array__array_neurongroup_I." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_static_array__array_neurongroup_I, _static_array__array_neurongroup_I, sizeof(double)*10, cudaMemcpyHostToDevice)
            );
    ifstream f_static_array__array_synapses_sources;
    f_static_array__array_synapses_sources.open("static_arrays/_static_array__array_synapses_sources", ios::in | ios::binary);
    if(f_static_array__array_synapses_sources.is_open())
    {
        f_static_array__array_synapses_sources.read(reinterpret_cast<char*>(_static_array__array_synapses_sources), 100*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_synapses_sources." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_static_array__array_synapses_sources, _static_array__array_synapses_sources, sizeof(int32_t)*100, cudaMemcpyHostToDevice)
            );
    ifstream f_static_array__array_synapses_targets;
    f_static_array__array_synapses_targets.open("static_arrays/_static_array__array_synapses_targets", ios::in | ios::binary);
    if(f_static_array__array_synapses_targets.is_open())
    {
        f_static_array__array_synapses_targets.read(reinterpret_cast<char*>(_static_array__array_synapses_targets), 100*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_synapses_targets." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_static_array__array_synapses_targets, _static_array__array_synapses_targets, sizeof(int32_t)*100, cudaMemcpyHostToDevice)
            );
}

void _write_arrays()
{
    using namespace brian;

    ofstream outfile__array_defaultclock_dt;
    outfile__array_defaultclock_dt.open("results/_array_defaultclock_dt_-2429358920718252942", ios::binary | ios::out);
    if(outfile__array_defaultclock_dt.is_open())
    {
        outfile__array_defaultclock_dt.write(reinterpret_cast<char*>(_array_defaultclock_dt), 1*sizeof(double));
        outfile__array_defaultclock_dt.close();
    } else
    {
        std::cout << "Error writing output file for _array_defaultclock_dt." << endl;
    }
    ofstream outfile__array_defaultclock_t;
    outfile__array_defaultclock_t.open("results/_array_defaultclock_t_8221117554015445948", ios::binary | ios::out);
    if(outfile__array_defaultclock_t.is_open())
    {
        outfile__array_defaultclock_t.write(reinterpret_cast<char*>(_array_defaultclock_t), 1*sizeof(double));
        outfile__array_defaultclock_t.close();
    } else
    {
        std::cout << "Error writing output file for _array_defaultclock_t." << endl;
    }
    ofstream outfile__array_defaultclock_timestep;
    outfile__array_defaultclock_timestep.open("results/_array_defaultclock_timestep_-902546129400883849", ios::binary | ios::out);
    if(outfile__array_defaultclock_timestep.is_open())
    {
        outfile__array_defaultclock_timestep.write(reinterpret_cast<char*>(_array_defaultclock_timestep), 1*sizeof(int64_t));
        outfile__array_defaultclock_timestep.close();
    } else
    {
        std::cout << "Error writing output file for _array_defaultclock_timestep." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_i, dev_array_neurongroup_i, sizeof(int32_t)*_num__array_neurongroup_i, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_i;
    outfile__array_neurongroup_i.open("results/_array_neurongroup_i_-2187955702463724779", ios::binary | ios::out);
    if(outfile__array_neurongroup_i.is_open())
    {
        outfile__array_neurongroup_i.write(reinterpret_cast<char*>(_array_neurongroup_i), 10*sizeof(int32_t));
        outfile__array_neurongroup_i.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_i." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_I, dev_array_neurongroup_I, sizeof(double)*_num__array_neurongroup_I, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_I;
    outfile__array_neurongroup_I.open("results/_array_neurongroup_I_7859714576964995343", ios::binary | ios::out);
    if(outfile__array_neurongroup_I.is_open())
    {
        outfile__array_neurongroup_I.write(reinterpret_cast<char*>(_array_neurongroup_I), 10*sizeof(double));
        outfile__array_neurongroup_I.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_I." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_Ibias, dev_array_neurongroup_Ibias, sizeof(double)*_num__array_neurongroup_Ibias, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_Ibias;
    outfile__array_neurongroup_Ibias.open("results/_array_neurongroup_Ibias_4456746137043582147", ios::binary | ios::out);
    if(outfile__array_neurongroup_Ibias.is_open())
    {
        outfile__array_neurongroup_Ibias.write(reinterpret_cast<char*>(_array_neurongroup_Ibias), 10*sizeof(double));
        outfile__array_neurongroup_Ibias.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_Ibias." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_Isyn, dev_array_neurongroup_Isyn, sizeof(double)*_num__array_neurongroup_Isyn, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_Isyn;
    outfile__array_neurongroup_Isyn.open("results/_array_neurongroup_Isyn_-2523609494053689006", ios::binary | ios::out);
    if(outfile__array_neurongroup_Isyn.is_open())
    {
        outfile__array_neurongroup_Isyn.write(reinterpret_cast<char*>(_array_neurongroup_Isyn), 10*sizeof(double));
        outfile__array_neurongroup_Isyn.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_Isyn." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_v, dev_array_neurongroup_v, sizeof(double)*_num__array_neurongroup_v, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_v;
    outfile__array_neurongroup_v.open("results/_array_neurongroup_v_-5915755877620378284", ios::binary | ios::out);
    if(outfile__array_neurongroup_v.is_open())
    {
        outfile__array_neurongroup_v.write(reinterpret_cast<char*>(_array_neurongroup_v), 10*sizeof(double));
        outfile__array_neurongroup_v.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_v." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_N, dev_array_synapses_N, sizeof(int32_t)*_num__array_synapses_N, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_N;
    outfile__array_synapses_N.open("results/_array_synapses_N_-5246794791579663891", ios::binary | ios::out);
    if(outfile__array_synapses_N.is_open())
    {
        outfile__array_synapses_N.write(reinterpret_cast<char*>(_array_synapses_N), 1*sizeof(int32_t));
        outfile__array_synapses_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_N." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_sources, dev_array_synapses_sources, sizeof(int32_t)*_num__array_synapses_sources, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_sources;
    outfile__array_synapses_sources.open("results/_array_synapses_sources_2315708605618965934", ios::binary | ios::out);
    if(outfile__array_synapses_sources.is_open())
    {
        outfile__array_synapses_sources.write(reinterpret_cast<char*>(_array_synapses_sources), 100*sizeof(int32_t));
        outfile__array_synapses_sources.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_sources." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_targets, dev_array_synapses_targets, sizeof(int32_t)*_num__array_synapses_targets, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_targets;
    outfile__array_synapses_targets.open("results/_array_synapses_targets_-5788833470487394847", ios::binary | ios::out);
    if(outfile__array_synapses_targets.is_open())
    {
        outfile__array_synapses_targets.write(reinterpret_cast<char*>(_array_synapses_targets), 100*sizeof(int32_t));
        outfile__array_synapses_targets.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_targets." << endl;
    }

    ofstream outfile__dynamic_array_synapses__synaptic_post;
    outfile__dynamic_array_synapses__synaptic_post.open("results/_dynamic_array_synapses__synaptic_post_-3350795012496696554", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses__synaptic_post.is_open())
    {
        outfile__dynamic_array_synapses__synaptic_post.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses__synaptic_post[0])), _dynamic_array_synapses__synaptic_post.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses__synaptic_post.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses__synaptic_post." << endl;
    }
    ofstream outfile__dynamic_array_synapses__synaptic_pre;
    outfile__dynamic_array_synapses__synaptic_pre.open("results/_dynamic_array_synapses__synaptic_pre_1038413857213702262", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses__synaptic_pre.is_open())
    {
        outfile__dynamic_array_synapses__synaptic_pre.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses__synaptic_pre[0])), _dynamic_array_synapses__synaptic_pre.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses__synaptic_pre.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses__synaptic_pre." << endl;
    }
    _dynamic_array_synapses_N_incoming = dev_dynamic_array_synapses_N_incoming;
    ofstream outfile__dynamic_array_synapses_N_incoming;
    outfile__dynamic_array_synapses_N_incoming.open("results/_dynamic_array_synapses_N_incoming_3090971508870193020", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_N_incoming.is_open())
    {
        outfile__dynamic_array_synapses_N_incoming.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_N_incoming[0])), _dynamic_array_synapses_N_incoming.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_N_incoming.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_N_incoming." << endl;
    }
    _dynamic_array_synapses_N_outgoing = dev_dynamic_array_synapses_N_outgoing;
    ofstream outfile__dynamic_array_synapses_N_outgoing;
    outfile__dynamic_array_synapses_N_outgoing.open("results/_dynamic_array_synapses_N_outgoing_-4618402319706929872", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_N_outgoing.is_open())
    {
        outfile__dynamic_array_synapses_N_outgoing.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_N_outgoing[0])), _dynamic_array_synapses_N_outgoing.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_N_outgoing.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_N_outgoing." << endl;
    }


    // Write last run info to disk
    ofstream outfile_last_run_info;
    outfile_last_run_info.open("results/last_run_info.txt", ios::out);
    if(outfile_last_run_info.is_open())
    {
        outfile_last_run_info << (Network::_last_run_time) << " " << (Network::_last_run_completed_fraction) << std::endl;
        outfile_last_run_info.close();
    } else
    {
        std::cout << "Error writing last run info to file." << std::endl;
    }
}


void _dealloc_arrays()
{
    using namespace brian;


    CUDA_SAFE_CALL(
            curandDestroyGenerator(curand_generator)
            );


    dev_dynamic_array_synapses__synaptic_post.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses__synaptic_post);
    _dynamic_array_synapses__synaptic_post.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses__synaptic_post);
    dev_dynamic_array_synapses__synaptic_pre.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses__synaptic_pre);
    _dynamic_array_synapses__synaptic_pre.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses__synaptic_pre);
    dev_dynamic_array_synapses_N_incoming.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_N_incoming);
    _dynamic_array_synapses_N_incoming.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_N_incoming);
    dev_dynamic_array_synapses_N_outgoing.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_N_outgoing);
    _dynamic_array_synapses_N_outgoing.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_N_outgoing);

    if(_array_defaultclock_dt!=0)
    {
        delete [] _array_defaultclock_dt;
        _array_defaultclock_dt = 0;
    }
    if(dev_array_defaultclock_dt!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_defaultclock_dt)
                );
        dev_array_defaultclock_dt = 0;
    }
    if(_array_defaultclock_t!=0)
    {
        delete [] _array_defaultclock_t;
        _array_defaultclock_t = 0;
    }
    if(dev_array_defaultclock_t!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_defaultclock_t)
                );
        dev_array_defaultclock_t = 0;
    }
    if(_array_defaultclock_timestep!=0)
    {
        delete [] _array_defaultclock_timestep;
        _array_defaultclock_timestep = 0;
    }
    if(dev_array_defaultclock_timestep!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_defaultclock_timestep)
                );
        dev_array_defaultclock_timestep = 0;
    }
    if(_array_neurongroup_i!=0)
    {
        delete [] _array_neurongroup_i;
        _array_neurongroup_i = 0;
    }
    if(dev_array_neurongroup_i!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_i)
                );
        dev_array_neurongroup_i = 0;
    }
    if(_array_neurongroup_I!=0)
    {
        delete [] _array_neurongroup_I;
        _array_neurongroup_I = 0;
    }
    if(dev_array_neurongroup_I!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_I)
                );
        dev_array_neurongroup_I = 0;
    }
    if(_array_neurongroup_Ibias!=0)
    {
        delete [] _array_neurongroup_Ibias;
        _array_neurongroup_Ibias = 0;
    }
    if(dev_array_neurongroup_Ibias!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_Ibias)
                );
        dev_array_neurongroup_Ibias = 0;
    }
    if(_array_neurongroup_Isyn!=0)
    {
        delete [] _array_neurongroup_Isyn;
        _array_neurongroup_Isyn = 0;
    }
    if(dev_array_neurongroup_Isyn!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_Isyn)
                );
        dev_array_neurongroup_Isyn = 0;
    }
    if(_array_neurongroup_v!=0)
    {
        delete [] _array_neurongroup_v;
        _array_neurongroup_v = 0;
    }
    if(dev_array_neurongroup_v!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_v)
                );
        dev_array_neurongroup_v = 0;
    }
    if(_array_synapses_N!=0)
    {
        delete [] _array_synapses_N;
        _array_synapses_N = 0;
    }
    if(dev_array_synapses_N!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_N)
                );
        dev_array_synapses_N = 0;
    }
    if(_array_synapses_sources!=0)
    {
        delete [] _array_synapses_sources;
        _array_synapses_sources = 0;
    }
    if(dev_array_synapses_sources!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_sources)
                );
        dev_array_synapses_sources = 0;
    }
    if(_array_synapses_targets!=0)
    {
        delete [] _array_synapses_targets;
        _array_synapses_targets = 0;
    }
    if(dev_array_synapses_targets!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_targets)
                );
        dev_array_synapses_targets = 0;
    }


    // static arrays
    if(_static_array__array_neurongroup_I!=0)
    {
        delete [] _static_array__array_neurongroup_I;
        _static_array__array_neurongroup_I = 0;
    }
    if(_static_array__array_synapses_sources!=0)
    {
        delete [] _static_array__array_synapses_sources;
        _static_array__array_synapses_sources = 0;
    }
    if(_static_array__array_synapses_targets!=0)
    {
        delete [] _static_array__array_synapses_targets;
        _static_array__array_synapses_targets = 0;
    }


}

