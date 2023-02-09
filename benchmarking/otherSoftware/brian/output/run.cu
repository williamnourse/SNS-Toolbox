#include<stdlib.h>
#include "brianlib/cuda_utils.h"
#include "objects.h"
#include<ctime>

#include "code_objects/synapses_synapses_create_array_codeobject.h"


void brian_start()
{
    _init_arrays();
    _load_arrays();
    srand(time(NULL));

    // Initialize clocks (link timestep and dt to the respective arrays)
}

void brian_end()
{
    _write_arrays();
    _dealloc_arrays();
}


