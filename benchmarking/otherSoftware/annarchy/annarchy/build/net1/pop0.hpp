/*
 *  ANNarchy-version: 4.7.2.2
 */
 #pragma once
#include "ANNarchy.h"

// host defines
extern double dt;
extern long int t;

// RNG - defined in ANNarchy.cu
extern unsigned long long global_seed;
extern void init_curand_states( int N, curandState* states, unsigned long long seed );







///////////////////////////////////////////////////////////////
// Main Structure for the population of id 0 (pop0)
///////////////////////////////////////////////////////////////
struct PopStruct0{
    int size; // Number of neurons
    bool _active; // Allows to shut down the whole population
    int max_delay; // Maximum number of steps to store for delayed synaptic transmission
    
    // CUDA launch configuration
    cudaStream_t stream;
    unsigned int _nb_blocks;
    unsigned int _threads_per_block;

    // Access functions used by cython wrapper
    int get_size() { return size; }
    void set_size(int s) { size  = s; }
    int get_max_delay() { return max_delay; }
    void set_max_delay(int d) { max_delay  = d; }
    bool is_active() { return _active; }
    void set_active(bool val) { _active = val; }


    // Neuron specific parameters and variables

    // Local attribute Cm
    std::vector< double > Cm;
    double *gpu_Cm;
    long int Cm_device_to_host;
    bool Cm_host_to_device;

    // Local attribute Gm
    std::vector< double > Gm;
    double *gpu_Gm;
    long int Gm_device_to_host;
    bool Gm_host_to_device;

    // Local attribute bias
    std::vector< double > bias;
    double *gpu_bias;
    long int bias_device_to_host;
    bool bias_host_to_device;

    // Local attribute Esyn
    std::vector< double > Esyn;
    double *gpu_Esyn;
    long int Esyn_device_to_host;
    bool Esyn_host_to_device;

    // Local attribute v
    std::vector< double > v;
    double *gpu_v;
    long int v_device_to_host;
    bool v_host_to_device;

    // Local attribute r
    std::vector< double > r;
    double *gpu_r;
    long int r_device_to_host;
    bool r_host_to_device;

    // Local attribute _sum_exc
    std::vector< double > _sum_exc;
    double *gpu__sum_exc;
    long int _sum_exc_device_to_host;
    bool _sum_exc_host_to_device;

    // Random numbers





    // Profiling


    // Access methods to the parameters and variables

    std::vector<double> get_local_attribute_all_double(std::string name) {

        // Local parameter Cm
        if ( name.compare("Cm") == 0 ) {
            if ( Cm_device_to_host < t ) device_to_host();
            return Cm;
        }

        // Local parameter Gm
        if ( name.compare("Gm") == 0 ) {
            if ( Gm_device_to_host < t ) device_to_host();
            return Gm;
        }

        // Local parameter bias
        if ( name.compare("bias") == 0 ) {
            if ( bias_device_to_host < t ) device_to_host();
            return bias;
        }

        // Local parameter Esyn
        if ( name.compare("Esyn") == 0 ) {
            if ( Esyn_device_to_host < t ) device_to_host();
            return Esyn;
        }

        // Local variable v
        if ( name.compare("v") == 0 ) {
            if ( v_device_to_host < t ) device_to_host();
            return v;
        }

        // Local variable r
        if ( name.compare("r") == 0 ) {
            if ( r_device_to_host < t ) device_to_host();
            return r;
        }

        // Local psp _sum_exc
        if ( name.compare("_sum_exc") == 0 ) {
            if ( _sum_exc_device_to_host < t ) device_to_host();
            return _sum_exc;
        }


        // should not happen
        std::cerr << "PopStruct0::get_local_attribute_all_double: " << name << " not found" << std::endl;
        return std::vector<double>();
    }

    double get_local_attribute_double(std::string name, int rk) {

        // Local parameter Cm
        if ( name.compare("Cm") == 0 ) {
            if ( Cm_device_to_host < t ) device_to_host();
            return Cm[rk];
        }

        // Local parameter Gm
        if ( name.compare("Gm") == 0 ) {
            if ( Gm_device_to_host < t ) device_to_host();
            return Gm[rk];
        }

        // Local parameter bias
        if ( name.compare("bias") == 0 ) {
            if ( bias_device_to_host < t ) device_to_host();
            return bias[rk];
        }

        // Local parameter Esyn
        if ( name.compare("Esyn") == 0 ) {
            if ( Esyn_device_to_host < t ) device_to_host();
            return Esyn[rk];
        }

        // Local variable v
        if ( name.compare("v") == 0 ) {
            if ( v_device_to_host < t ) device_to_host();
            return v[rk];
        }

        // Local variable r
        if ( name.compare("r") == 0 ) {
            if ( r_device_to_host < t ) device_to_host();
            return r[rk];
        }

        // Local psp _sum_exc
        if ( name.compare("_sum_exc") == 0 ) {
            if ( _sum_exc_device_to_host < t ) device_to_host();
            return _sum_exc[rk];
        }


        // should not happen
        std::cerr << "PopStruct0::get_local_attribute_double: " << name << " not found" << std::endl;
        return static_cast<double>(0.0);
    }

    void set_local_attribute_all_double(std::string name, std::vector<double> value) {

        // Local parameter Cm
        if ( name.compare("Cm") == 0 ) {
            Cm = value;
            Cm_host_to_device = true;
            return;
        }

        // Local parameter Gm
        if ( name.compare("Gm") == 0 ) {
            Gm = value;
            Gm_host_to_device = true;
            return;
        }

        // Local parameter bias
        if ( name.compare("bias") == 0 ) {
            bias = value;
            bias_host_to_device = true;
            return;
        }

        // Local parameter Esyn
        if ( name.compare("Esyn") == 0 ) {
            Esyn = value;
            Esyn_host_to_device = true;
            return;
        }

        // Local variable v
        if ( name.compare("v") == 0 ) {
            v = value;
            v_host_to_device = true;
            return;
        }

        // Local variable r
        if ( name.compare("r") == 0 ) {
            r = value;
            r_host_to_device = true;
            return;
        }

        // Local psp _sum_exc
        if ( name.compare("_sum_exc") == 0 ) {
            _sum_exc = value;
            _sum_exc_host_to_device = true;
            return;
        }


        // should not happen
        std::cerr << "PopStruct0::set_local_attribute_all_double: " << name << " not found" << std::endl;
    }

    void set_local_attribute_double(std::string name, int rk, double value) {

        // Local parameter Cm
        if ( name.compare("Cm") == 0 ) {
            Cm[rk] = value;
            Cm_host_to_device = true;
            return;
        }

        // Local parameter Gm
        if ( name.compare("Gm") == 0 ) {
            Gm[rk] = value;
            Gm_host_to_device = true;
            return;
        }

        // Local parameter bias
        if ( name.compare("bias") == 0 ) {
            bias[rk] = value;
            bias_host_to_device = true;
            return;
        }

        // Local parameter Esyn
        if ( name.compare("Esyn") == 0 ) {
            Esyn[rk] = value;
            Esyn_host_to_device = true;
            return;
        }

        // Local variable v
        if ( name.compare("v") == 0 ) {
            v[rk] = value;
            v_host_to_device = true;
            return;
        }

        // Local variable r
        if ( name.compare("r") == 0 ) {
            r[rk] = value;
            r_host_to_device = true;
            return;
        }

        // Local psp _sum_exc
        if ( name.compare("_sum_exc") == 0 ) {
            _sum_exc[rk] = value;
            _sum_exc_host_to_device = true;
            return;
        }


        // should not happen
        std::cerr << "PopStruct0::set_local_attribute_double: " << name << " not found" << std::endl;
    }



    // Method called to initialize the data structures
    void init_population() {
    #ifdef _DEBUG
        std::cout << "PopStruct0::init_population()" << std::endl;
    #endif
        _active = true;

        //
        // Launch configuration
        _threads_per_block = 128;
        _nb_blocks = static_cast<unsigned int>(ceil( static_cast<double>(size) / static_cast<double>(_threads_per_block) ) );
        _nb_blocks = std::min<unsigned int>(_nb_blocks, 65535);

        //
        // Model equations/parameters

        // Local parameter Cm
        Cm = std::vector<double>(size, 0.0);
        cudaMalloc(&gpu_Cm, size * sizeof(double));
        cudaMemcpy(gpu_Cm, Cm.data(), size * sizeof(double), cudaMemcpyHostToDevice);
    #ifdef _DEBUG
        cudaError_t err_Cm = cudaGetLastError();
        if ( err_Cm != cudaSuccess )
            std::cout << "    allocation of Cm failed: " << cudaGetErrorString(err_Cm) << std::endl;
    #endif
        // memory transfer flags
        Cm_host_to_device = false;
        Cm_device_to_host = t;

        // Local parameter Gm
        Gm = std::vector<double>(size, 0.0);
        cudaMalloc(&gpu_Gm, size * sizeof(double));
        cudaMemcpy(gpu_Gm, Gm.data(), size * sizeof(double), cudaMemcpyHostToDevice);
    #ifdef _DEBUG
        cudaError_t err_Gm = cudaGetLastError();
        if ( err_Gm != cudaSuccess )
            std::cout << "    allocation of Gm failed: " << cudaGetErrorString(err_Gm) << std::endl;
    #endif
        // memory transfer flags
        Gm_host_to_device = false;
        Gm_device_to_host = t;

        // Local parameter bias
        bias = std::vector<double>(size, 0.0);
        cudaMalloc(&gpu_bias, size * sizeof(double));
        cudaMemcpy(gpu_bias, bias.data(), size * sizeof(double), cudaMemcpyHostToDevice);
    #ifdef _DEBUG
        cudaError_t err_bias = cudaGetLastError();
        if ( err_bias != cudaSuccess )
            std::cout << "    allocation of bias failed: " << cudaGetErrorString(err_bias) << std::endl;
    #endif
        // memory transfer flags
        bias_host_to_device = false;
        bias_device_to_host = t;

        // Local parameter Esyn
        Esyn = std::vector<double>(size, 0.0);
        cudaMalloc(&gpu_Esyn, size * sizeof(double));
        cudaMemcpy(gpu_Esyn, Esyn.data(), size * sizeof(double), cudaMemcpyHostToDevice);
    #ifdef _DEBUG
        cudaError_t err_Esyn = cudaGetLastError();
        if ( err_Esyn != cudaSuccess )
            std::cout << "    allocation of Esyn failed: " << cudaGetErrorString(err_Esyn) << std::endl;
    #endif
        // memory transfer flags
        Esyn_host_to_device = false;
        Esyn_device_to_host = t;

        // Local variable v
        v = std::vector<double>(size, 0.0);
        cudaMalloc(&gpu_v, size * sizeof(double));
        cudaMemcpy(gpu_v, v.data(), size * sizeof(double), cudaMemcpyHostToDevice);
    #ifdef _DEBUG
        cudaError_t err_v = cudaGetLastError();
        if ( err_v != cudaSuccess )
            std::cout << "    allocation of v failed: " << cudaGetErrorString(err_v) << std::endl;
    #endif
        // memory transfer flags
        v_host_to_device = false;
        v_device_to_host = t;

        // Local variable r
        r = std::vector<double>(size, 0.0);
        cudaMalloc(&gpu_r, size * sizeof(double));
        cudaMemcpy(gpu_r, r.data(), size * sizeof(double), cudaMemcpyHostToDevice);
    #ifdef _DEBUG
        cudaError_t err_r = cudaGetLastError();
        if ( err_r != cudaSuccess )
            std::cout << "    allocation of r failed: " << cudaGetErrorString(err_r) << std::endl;
    #endif
        // memory transfer flags
        r_host_to_device = false;
        r_device_to_host = t;

        // Local psp _sum_exc
        _sum_exc = std::vector<double>(size, 0.0);
        cudaMalloc(&gpu__sum_exc, size * sizeof(double));
        cudaMemcpy(gpu__sum_exc, _sum_exc.data(), size * sizeof(double), cudaMemcpyHostToDevice);
    #ifdef _DEBUG
        cudaError_t err__sum_exc = cudaGetLastError();
        if ( err__sum_exc != cudaSuccess )
            std::cout << "    allocation of _sum_exc failed: " << cudaGetErrorString(err__sum_exc) << std::endl;
    #endif
        // memory transfer flags
        _sum_exc_host_to_device = false;
        _sum_exc_device_to_host = t;






    }

    // Method called to reset the population
    void reset() {



        // read-back flags: variables
        v_device_to_host = 0;
        r_device_to_host = 0;
        
        // read-back flags: targets
        _sum_exc_device_to_host = 0;
        
    }

    // Method to draw new random numbers
    void update_rng() {

    }

    // Method to enqueue output variables in case outgoing projections have non-zero delay
    void update_delay() {

    }

    // Method to dynamically change the size of the queue for delayed variables
    void update_max_delay(int value) {

    }

    // Main method to update neural variables
    void update() {

    }

    // Mean-firing rate computed on host
    void update_FR() {

    }

    // Stop condition
    

    // Memory transfers
    void host_to_device() {

    // host to device transfers for pop0
        // v: local
        if( v_host_to_device )
        {
        #ifdef _DEBUG
            std::cout << "HtoD v ( pop0 )" << std::endl;
        #endif
            cudaMemcpy( gpu_v, v.data(), size * sizeof(double), cudaMemcpyHostToDevice);
            v_host_to_device = false;

        #ifdef _DEBUG
            cudaError_t err_v = cudaGetLastError();
            if ( err_v != cudaSuccess )
                std::cout << "  error: " << cudaGetErrorString(err_v) << std::endl;
        #endif
        }
    
        // r: local
        if( r_host_to_device )
        {
        #ifdef _DEBUG
            std::cout << "HtoD r ( pop0 )" << std::endl;
        #endif
            cudaMemcpy( gpu_r, r.data(), size * sizeof(double), cudaMemcpyHostToDevice);
            r_host_to_device = false;

        #ifdef _DEBUG
            cudaError_t err_r = cudaGetLastError();
            if ( err_r != cudaSuccess )
                std::cout << "  error: " << cudaGetErrorString(err_r) << std::endl;
        #endif
        }
    
        // Cm: local
        if( Cm_host_to_device )
        {
        #ifdef _DEBUG
            std::cout << "HtoD Cm ( pop0 )" << std::endl;
        #endif
            cudaMemcpy( gpu_Cm, Cm.data(), size * sizeof(double), cudaMemcpyHostToDevice);
            Cm_host_to_device = false;

        #ifdef _DEBUG
            cudaError_t err_Cm = cudaGetLastError();
            if ( err_Cm != cudaSuccess )
                std::cout << "  error: " << cudaGetErrorString(err_Cm) << std::endl;
        #endif
        }
    
        // Gm: local
        if( Gm_host_to_device )
        {
        #ifdef _DEBUG
            std::cout << "HtoD Gm ( pop0 )" << std::endl;
        #endif
            cudaMemcpy( gpu_Gm, Gm.data(), size * sizeof(double), cudaMemcpyHostToDevice);
            Gm_host_to_device = false;

        #ifdef _DEBUG
            cudaError_t err_Gm = cudaGetLastError();
            if ( err_Gm != cudaSuccess )
                std::cout << "  error: " << cudaGetErrorString(err_Gm) << std::endl;
        #endif
        }
    
        // bias: local
        if( bias_host_to_device )
        {
        #ifdef _DEBUG
            std::cout << "HtoD bias ( pop0 )" << std::endl;
        #endif
            cudaMemcpy( gpu_bias, bias.data(), size * sizeof(double), cudaMemcpyHostToDevice);
            bias_host_to_device = false;

        #ifdef _DEBUG
            cudaError_t err_bias = cudaGetLastError();
            if ( err_bias != cudaSuccess )
                std::cout << "  error: " << cudaGetErrorString(err_bias) << std::endl;
        #endif
        }
    
        // Esyn: local
        if( Esyn_host_to_device )
        {
        #ifdef _DEBUG
            std::cout << "HtoD Esyn ( pop0 )" << std::endl;
        #endif
            cudaMemcpy( gpu_Esyn, Esyn.data(), size * sizeof(double), cudaMemcpyHostToDevice);
            Esyn_host_to_device = false;

        #ifdef _DEBUG
            cudaError_t err_Esyn = cudaGetLastError();
            if ( err_Esyn != cudaSuccess )
                std::cout << "  error: " << cudaGetErrorString(err_Esyn) << std::endl;
        #endif
        }
    
        // _sum_exc: local
        if( _sum_exc_host_to_device )
        {
        #ifdef _DEBUG
            std::cout << "HtoD _sum_exc ( pop0 )" << std::endl;
        #endif
            cudaMemcpy( gpu__sum_exc, _sum_exc.data(), size * sizeof(double), cudaMemcpyHostToDevice);
            _sum_exc_host_to_device = false;

        #ifdef _DEBUG
            cudaError_t err__sum_exc = cudaGetLastError();
            if ( err__sum_exc != cudaSuccess )
                std::cout << "  error: " << cudaGetErrorString(err__sum_exc) << std::endl;
        #endif
        }
    
    }

    void device_to_host() {

    // device to host transfers for pop0

        // v: local
        if( v_device_to_host < t ) {
        #ifdef _DEBUG
            std::cout << "DtoH: v ( pop0 )" << std::endl;
        #endif
            cudaMemcpy( v.data(),  gpu_v, size * sizeof(double), cudaMemcpyDeviceToHost);
        #ifdef _DEBUG
            cudaError_t err_v = cudaGetLastError();
            if ( err_v != cudaSuccess )
                std::cout << "  error: " << cudaGetErrorString(err_v) << std::endl;
        #endif
            v_device_to_host = t;
        }
    
        // r: local
        if( r_device_to_host < t ) {
        #ifdef _DEBUG
            std::cout << "DtoH: r ( pop0 )" << std::endl;
        #endif
            cudaMemcpy( r.data(),  gpu_r, size * sizeof(double), cudaMemcpyDeviceToHost);
        #ifdef _DEBUG
            cudaError_t err_r = cudaGetLastError();
            if ( err_r != cudaSuccess )
                std::cout << "  error: " << cudaGetErrorString(err_r) << std::endl;
        #endif
            r_device_to_host = t;
        }
    
        // _sum_exc: local
        if( _sum_exc_device_to_host < t ) {
        #ifdef _DEBUG
            std::cout << "DtoH: _sum_exc ( pop0 )" << std::endl;
        #endif
            cudaMemcpy( _sum_exc.data(),  gpu__sum_exc, size * sizeof(double), cudaMemcpyDeviceToHost);
        #ifdef _DEBUG
            cudaError_t err__sum_exc = cudaGetLastError();
            if ( err__sum_exc != cudaSuccess )
                std::cout << "  error: " << cudaGetErrorString(err__sum_exc) << std::endl;
        #endif
            _sum_exc_device_to_host = t;
        }
    
    }

    // Memory Management: track memory consumption
    long int size_in_bytes() {
        long int size_in_bytes = 0;
        // Parameters
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * Cm.capacity();	// Cm
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * Gm.capacity();	// Gm
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * bias.capacity();	// bias
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * Esyn.capacity();	// Esyn
        // Variables
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * v.capacity();	// v
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * r.capacity();	// r
        // RNGs
        
        return size_in_bytes;
    }

    // Memory Management: clear container
    void clear() {
        // Variables
        v.clear();
        v.shrink_to_fit();
        r.clear();
        r.shrink_to_fit();
        
        /* Free device variables */
        // parameters
        cudaFree(gpu_Cm); 
        cudaFree(gpu_Gm); 
        cudaFree(gpu_bias); 
        cudaFree(gpu_Esyn); 
        
        // variables
        cudaFree(gpu_v); 
        cudaFree(gpu_r); 
        
        // delayed attributes
        
        // RNGs
        
        // targets
        cudaFree(gpu__sum_exc); 
        
    #ifdef _DEBUG
        cudaError_t err_clear = cudaGetLastError();
        if ( err_clear != cudaSuccess )
            std::cout << "Pop0::clear() - cudaFree: " << cudaGetErrorString(err_clear) << std::endl;
    #endif

    }
};
