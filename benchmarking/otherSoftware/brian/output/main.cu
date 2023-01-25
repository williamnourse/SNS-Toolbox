#include <stdlib.h>
#include "objects.h"
#include <ctime>
#include <time.h>
#include "run.h"
#include "brianlib/common_math.h"
#include "brianlib/cuda_utils.h"
#include "rand.h"

#include "code_objects/synapses_synapses_create_array_codeobject.h"


#include <iostream>
#include <fstream>
#include <string>
#include "cuda_profiler_api.h"




int main(int argc, char **argv)
{
        


    // seed variable set in Python through brian2.seed() calls can use this
    // variable (see device.py CUDAStandaloneDevice.generate_main_source())
    unsigned long long seed;

    //const std::clock_t _start_time = std::clock();

    CUDA_SAFE_CALL(
            cudaSetDevice(0)
            );

    cudaDeviceProp props;
    CUDA_SAFE_CALL(
            cudaGetDeviceProperties(&props, 0)
            );
    size_t limit = 128 * 1024 * 1024;
    CUDA_SAFE_CALL(
            cudaDeviceSetLimit(cudaLimitMallocHeapSize, limit)
            );
    CUDA_SAFE_CALL(
            cudaDeviceSynchronize()
            );

    //const double _run_time2 = (double)(std::clock() -_start_time)/CLOCKS_PER_SEC;
    //printf("INFO: setting cudaDevice stuff took %f seconds\n", _run_time2);

    brian_start();

        


    //const std::clock_t _start_time3 = std::clock();
    {
        using namespace brian;

                
        _array_defaultclock_dt[0] = 0.0001;
        _array_defaultclock_dt[0] = 0.0001;
        _array_defaultclock_dt[0] = 0.0001;
        _array_defaultclock_dt[0] = 0.0001;
        for(int i=0; i<_num__static_array__array_neurongroup_I; i++)
        {
            _array_neurongroup_I[i] = _static_array__array_neurongroup_I[i];
        }
            CUDA_SAFE_CALL(
                cudaMemcpy(
                    dev_array_neurongroup_I,
                    &_array_neurongroup_I[0],
                    sizeof(_array_neurongroup_I[0])*_num__array_neurongroup_I,
                    cudaMemcpyHostToDevice
                )
            );
        for(int i=0; i<_num__static_array__array_synapses_sources; i++)
        {
            _array_synapses_sources[i] = _static_array__array_synapses_sources[i];
        }
            CUDA_SAFE_CALL(
                cudaMemcpy(
                    dev_array_synapses_sources,
                    &_array_synapses_sources[0],
                    sizeof(_array_synapses_sources[0])*_num__array_synapses_sources,
                    cudaMemcpyHostToDevice
                )
            );
        for(int i=0; i<_num__static_array__array_synapses_targets; i++)
        {
            _array_synapses_targets[i] = _static_array__array_synapses_targets[i];
        }
            CUDA_SAFE_CALL(
                cudaMemcpy(
                    dev_array_synapses_targets,
                    &_array_synapses_targets[0],
                    sizeof(_array_synapses_targets[0])*_num__array_synapses_targets,
                    cudaMemcpyHostToDevice
                )
            );
        _run_synapses_synapses_create_array_codeobject();

    }

    //const double _run_time3 = (double)(std::clock() -_start_time3)/CLOCKS_PER_SEC;
    //printf("INFO: main_lines took %f seconds\n", _run_time3);

        

    brian_end();
        


    // Profiling
    //const double _run_time = (double)(std::clock() -_start_time)/CLOCKS_PER_SEC;
    //printf("INFO: main function took %f seconds\n", _run_time);

    return 0;
}