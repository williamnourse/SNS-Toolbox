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


    // Structures for managing spikes
    std::vector<long int> last_spike;
    long int* gpu_last_spike;
    std::vector<int> spiked;
    int* gpu_spiked;
    unsigned int spike_count;
    unsigned int* gpu_spike_count;

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

    // Local attribute tau
    std::vector< double > tau;
    double *gpu_tau;
    long int tau_device_to_host;
    bool tau_host_to_device;

    // Local attribute To
    std::vector< double > To;
    double *gpu_To;
    long int To_device_to_host;
    bool To_host_to_device;

    // Local attribute m
    std::vector< double > m;
    double *gpu_m;
    long int m_device_to_host;
    bool m_host_to_device;

    // Local attribute tau_inh
    std::vector< double > tau_inh;
    double *gpu_tau_inh;
    long int tau_inh_device_to_host;
    bool tau_inh_host_to_device;

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

    // Local attribute T
    std::vector< double > T;
    double *gpu_T;
    long int T_device_to_host;
    bool T_host_to_device;

    // Local attribute g_inh
    std::vector< double > g_inh;
    double *gpu_g_inh;
    long int g_inh_device_to_host;
    bool g_inh_host_to_device;

    // Local attribute r
    std::vector< double > r;
    double *gpu_r;
    long int r_device_to_host;
    bool r_host_to_device;

    // Random numbers



    // Mean Firing rate
    std::vector< std::queue<long int> > _spike_history;
    long int _mean_fr_window;
    double _mean_fr_rate;
    void compute_firing_rate( double window){
        if(window>0.0){
            _mean_fr_window = int(window/dt);
            _mean_fr_rate = double(1000./double(window));
            if (_spike_history.empty())
                _spike_history = std::vector< std::queue<long int> >(size, std::queue<long int>());
        }
    };


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

        // Local parameter tau
        if ( name.compare("tau") == 0 ) {
            if ( tau_device_to_host < t ) device_to_host();
            return tau;
        }

        // Local parameter To
        if ( name.compare("To") == 0 ) {
            if ( To_device_to_host < t ) device_to_host();
            return To;
        }

        // Local parameter m
        if ( name.compare("m") == 0 ) {
            if ( m_device_to_host < t ) device_to_host();
            return m;
        }

        // Local parameter tau_inh
        if ( name.compare("tau_inh") == 0 ) {
            if ( tau_inh_device_to_host < t ) device_to_host();
            return tau_inh;
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

        // Local variable T
        if ( name.compare("T") == 0 ) {
            if ( T_device_to_host < t ) device_to_host();
            return T;
        }

        // Local variable g_inh
        if ( name.compare("g_inh") == 0 ) {
            if ( g_inh_device_to_host < t ) device_to_host();
            return g_inh;
        }

        // Local variable r
        if ( name.compare("r") == 0 ) {
            if ( r_device_to_host < t ) device_to_host();
            return r;
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

        // Local parameter tau
        if ( name.compare("tau") == 0 ) {
            if ( tau_device_to_host < t ) device_to_host();
            return tau[rk];
        }

        // Local parameter To
        if ( name.compare("To") == 0 ) {
            if ( To_device_to_host < t ) device_to_host();
            return To[rk];
        }

        // Local parameter m
        if ( name.compare("m") == 0 ) {
            if ( m_device_to_host < t ) device_to_host();
            return m[rk];
        }

        // Local parameter tau_inh
        if ( name.compare("tau_inh") == 0 ) {
            if ( tau_inh_device_to_host < t ) device_to_host();
            return tau_inh[rk];
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

        // Local variable T
        if ( name.compare("T") == 0 ) {
            if ( T_device_to_host < t ) device_to_host();
            return T[rk];
        }

        // Local variable g_inh
        if ( name.compare("g_inh") == 0 ) {
            if ( g_inh_device_to_host < t ) device_to_host();
            return g_inh[rk];
        }

        // Local variable r
        if ( name.compare("r") == 0 ) {
            if ( r_device_to_host < t ) device_to_host();
            return r[rk];
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

        // Local parameter tau
        if ( name.compare("tau") == 0 ) {
            tau = value;
            tau_host_to_device = true;
            return;
        }

        // Local parameter To
        if ( name.compare("To") == 0 ) {
            To = value;
            To_host_to_device = true;
            return;
        }

        // Local parameter m
        if ( name.compare("m") == 0 ) {
            m = value;
            m_host_to_device = true;
            return;
        }

        // Local parameter tau_inh
        if ( name.compare("tau_inh") == 0 ) {
            tau_inh = value;
            tau_inh_host_to_device = true;
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

        // Local variable T
        if ( name.compare("T") == 0 ) {
            T = value;
            T_host_to_device = true;
            return;
        }

        // Local variable g_inh
        if ( name.compare("g_inh") == 0 ) {
            g_inh = value;
            g_inh_host_to_device = true;
            return;
        }

        // Local variable r
        if ( name.compare("r") == 0 ) {
            r = value;
            r_host_to_device = true;
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

        // Local parameter tau
        if ( name.compare("tau") == 0 ) {
            tau[rk] = value;
            tau_host_to_device = true;
            return;
        }

        // Local parameter To
        if ( name.compare("To") == 0 ) {
            To[rk] = value;
            To_host_to_device = true;
            return;
        }

        // Local parameter m
        if ( name.compare("m") == 0 ) {
            m[rk] = value;
            m_host_to_device = true;
            return;
        }

        // Local parameter tau_inh
        if ( name.compare("tau_inh") == 0 ) {
            tau_inh[rk] = value;
            tau_inh_host_to_device = true;
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

        // Local variable T
        if ( name.compare("T") == 0 ) {
            T[rk] = value;
            T_host_to_device = true;
            return;
        }

        // Local variable g_inh
        if ( name.compare("g_inh") == 0 ) {
            g_inh[rk] = value;
            g_inh_host_to_device = true;
            return;
        }

        // Local variable r
        if ( name.compare("r") == 0 ) {
            r[rk] = value;
            r_host_to_device = true;
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

        // Local parameter tau
        tau = std::vector<double>(size, 0.0);
        cudaMalloc(&gpu_tau, size * sizeof(double));
        cudaMemcpy(gpu_tau, tau.data(), size * sizeof(double), cudaMemcpyHostToDevice);
    #ifdef _DEBUG
        cudaError_t err_tau = cudaGetLastError();
        if ( err_tau != cudaSuccess )
            std::cout << "    allocation of tau failed: " << cudaGetErrorString(err_tau) << std::endl;
    #endif
        // memory transfer flags
        tau_host_to_device = false;
        tau_device_to_host = t;

        // Local parameter To
        To = std::vector<double>(size, 0.0);
        cudaMalloc(&gpu_To, size * sizeof(double));
        cudaMemcpy(gpu_To, To.data(), size * sizeof(double), cudaMemcpyHostToDevice);
    #ifdef _DEBUG
        cudaError_t err_To = cudaGetLastError();
        if ( err_To != cudaSuccess )
            std::cout << "    allocation of To failed: " << cudaGetErrorString(err_To) << std::endl;
    #endif
        // memory transfer flags
        To_host_to_device = false;
        To_device_to_host = t;

        // Local parameter m
        m = std::vector<double>(size, 0.0);
        cudaMalloc(&gpu_m, size * sizeof(double));
        cudaMemcpy(gpu_m, m.data(), size * sizeof(double), cudaMemcpyHostToDevice);
    #ifdef _DEBUG
        cudaError_t err_m = cudaGetLastError();
        if ( err_m != cudaSuccess )
            std::cout << "    allocation of m failed: " << cudaGetErrorString(err_m) << std::endl;
    #endif
        // memory transfer flags
        m_host_to_device = false;
        m_device_to_host = t;

        // Local parameter tau_inh
        tau_inh = std::vector<double>(size, 0.0);
        cudaMalloc(&gpu_tau_inh, size * sizeof(double));
        cudaMemcpy(gpu_tau_inh, tau_inh.data(), size * sizeof(double), cudaMemcpyHostToDevice);
    #ifdef _DEBUG
        cudaError_t err_tau_inh = cudaGetLastError();
        if ( err_tau_inh != cudaSuccess )
            std::cout << "    allocation of tau_inh failed: " << cudaGetErrorString(err_tau_inh) << std::endl;
    #endif
        // memory transfer flags
        tau_inh_host_to_device = false;
        tau_inh_device_to_host = t;

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

        // Local variable T
        T = std::vector<double>(size, 0.0);
        cudaMalloc(&gpu_T, size * sizeof(double));
        cudaMemcpy(gpu_T, T.data(), size * sizeof(double), cudaMemcpyHostToDevice);
    #ifdef _DEBUG
        cudaError_t err_T = cudaGetLastError();
        if ( err_T != cudaSuccess )
            std::cout << "    allocation of T failed: " << cudaGetErrorString(err_T) << std::endl;
    #endif
        // memory transfer flags
        T_host_to_device = false;
        T_device_to_host = t;

        // Local variable g_inh
        g_inh = std::vector<double>(size, 0.0);
        cudaMalloc(&gpu_g_inh, size * sizeof(double));
        cudaMemcpy(gpu_g_inh, g_inh.data(), size * sizeof(double), cudaMemcpyHostToDevice);
    #ifdef _DEBUG
        cudaError_t err_g_inh = cudaGetLastError();
        if ( err_g_inh != cudaSuccess )
            std::cout << "    allocation of g_inh failed: " << cudaGetErrorString(err_g_inh) << std::endl;
    #endif
        // memory transfer flags
        g_inh_host_to_device = false;
        g_inh_device_to_host = t;

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


        // Spiking variables
        spiked = std::vector<int>(size, 0);
        cudaMalloc((void**)&gpu_spiked, size * sizeof(int));
        cudaMemcpy(gpu_spiked, spiked.data(), size * sizeof(int), cudaMemcpyHostToDevice);

        last_spike = std::vector<long int>(size, -10000L);
        cudaMalloc((void**)&gpu_last_spike, size * sizeof(long int));
        cudaMemcpy(gpu_last_spike, last_spike.data(), size * sizeof(long int), cudaMemcpyHostToDevice);

        spike_count = 0;
        cudaMalloc((void**)&gpu_spike_count, sizeof(unsigned int));
        cudaMemcpy(gpu_spike_count, &spike_count, sizeof(unsigned int), cudaMemcpyHostToDevice);



        // Mean Firing Rate
        _spike_history = std::vector< std::queue<long int> >();
        _mean_fr_window = 0;
        _mean_fr_rate = 1.0;


    }

    // Method called to reset the population
    void reset() {

        spiked = std::vector<int>(size, 0);
        last_spike.clear();
        last_spike = std::vector<long int>(size, -10000L);
        spike_count = 0;

        // Mean Firing Rate
        for (auto it = _spike_history.begin(); it != _spike_history.end(); it++) {
            if (!it->empty()) {
                auto empty_queue = std::queue<long int>();
                it->swap(empty_queue);
            }
        }



        // read-back flags: variables
        v_device_to_host = 0;
        T_device_to_host = 0;
        g_inh_device_to_host = 0;
        r_device_to_host = 0;
        
        // read-back flags: targets
        
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

        if ( _mean_fr_window > 0) {
            // Update the queues
            r_host_to_device = false;

            for ( int i = 0; i < spike_count; i++ ) {
                _spike_history[spiked[i]].push(t);
                r_host_to_device = true; // the queue changed the length
            }

            // Recalculate the mean firing rate
            for (int i = 0; i < size; i++ ) {
                while((_spike_history[i].size() != 0)&&(_spike_history[i].front() <= t - _mean_fr_window)){
                    _spike_history[i].pop(); // Suppress spikes outside the window
                    r_host_to_device = true; // the queue changed the length
                }
                r[i] = _mean_fr_rate * float(_spike_history[i].size());
            }

            // transfer to device
            if ( r_host_to_device ) {
                cudaMemcpy(gpu_r, r.data(), size * sizeof(double), cudaMemcpyHostToDevice);
                r_host_to_device = false;
            }
        }

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
    
        // T: local
        if( T_host_to_device )
        {
        #ifdef _DEBUG
            std::cout << "HtoD T ( pop0 )" << std::endl;
        #endif
            cudaMemcpy( gpu_T, T.data(), size * sizeof(double), cudaMemcpyHostToDevice);
            T_host_to_device = false;

        #ifdef _DEBUG
            cudaError_t err_T = cudaGetLastError();
            if ( err_T != cudaSuccess )
                std::cout << "  error: " << cudaGetErrorString(err_T) << std::endl;
        #endif
        }
    
        // g_inh: local
        if( g_inh_host_to_device )
        {
        #ifdef _DEBUG
            std::cout << "HtoD g_inh ( pop0 )" << std::endl;
        #endif
            cudaMemcpy( gpu_g_inh, g_inh.data(), size * sizeof(double), cudaMemcpyHostToDevice);
            g_inh_host_to_device = false;

        #ifdef _DEBUG
            cudaError_t err_g_inh = cudaGetLastError();
            if ( err_g_inh != cudaSuccess )
                std::cout << "  error: " << cudaGetErrorString(err_g_inh) << std::endl;
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
    
        // tau: local
        if( tau_host_to_device )
        {
        #ifdef _DEBUG
            std::cout << "HtoD tau ( pop0 )" << std::endl;
        #endif
            cudaMemcpy( gpu_tau, tau.data(), size * sizeof(double), cudaMemcpyHostToDevice);
            tau_host_to_device = false;

        #ifdef _DEBUG
            cudaError_t err_tau = cudaGetLastError();
            if ( err_tau != cudaSuccess )
                std::cout << "  error: " << cudaGetErrorString(err_tau) << std::endl;
        #endif
        }
    
        // To: local
        if( To_host_to_device )
        {
        #ifdef _DEBUG
            std::cout << "HtoD To ( pop0 )" << std::endl;
        #endif
            cudaMemcpy( gpu_To, To.data(), size * sizeof(double), cudaMemcpyHostToDevice);
            To_host_to_device = false;

        #ifdef _DEBUG
            cudaError_t err_To = cudaGetLastError();
            if ( err_To != cudaSuccess )
                std::cout << "  error: " << cudaGetErrorString(err_To) << std::endl;
        #endif
        }
    
        // m: local
        if( m_host_to_device )
        {
        #ifdef _DEBUG
            std::cout << "HtoD m ( pop0 )" << std::endl;
        #endif
            cudaMemcpy( gpu_m, m.data(), size * sizeof(double), cudaMemcpyHostToDevice);
            m_host_to_device = false;

        #ifdef _DEBUG
            cudaError_t err_m = cudaGetLastError();
            if ( err_m != cudaSuccess )
                std::cout << "  error: " << cudaGetErrorString(err_m) << std::endl;
        #endif
        }
    
        // tau_inh: local
        if( tau_inh_host_to_device )
        {
        #ifdef _DEBUG
            std::cout << "HtoD tau_inh ( pop0 )" << std::endl;
        #endif
            cudaMemcpy( gpu_tau_inh, tau_inh.data(), size * sizeof(double), cudaMemcpyHostToDevice);
            tau_inh_host_to_device = false;

        #ifdef _DEBUG
            cudaError_t err_tau_inh = cudaGetLastError();
            if ( err_tau_inh != cudaSuccess )
                std::cout << "  error: " << cudaGetErrorString(err_tau_inh) << std::endl;
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
    
        // T: local
        if( T_device_to_host < t ) {
        #ifdef _DEBUG
            std::cout << "DtoH: T ( pop0 )" << std::endl;
        #endif
            cudaMemcpy( T.data(),  gpu_T, size * sizeof(double), cudaMemcpyDeviceToHost);
        #ifdef _DEBUG
            cudaError_t err_T = cudaGetLastError();
            if ( err_T != cudaSuccess )
                std::cout << "  error: " << cudaGetErrorString(err_T) << std::endl;
        #endif
            T_device_to_host = t;
        }
    
        // g_inh: local
        if( g_inh_device_to_host < t ) {
        #ifdef _DEBUG
            std::cout << "DtoH: g_inh ( pop0 )" << std::endl;
        #endif
            cudaMemcpy( g_inh.data(),  gpu_g_inh, size * sizeof(double), cudaMemcpyDeviceToHost);
        #ifdef _DEBUG
            cudaError_t err_g_inh = cudaGetLastError();
            if ( err_g_inh != cudaSuccess )
                std::cout << "  error: " << cudaGetErrorString(err_g_inh) << std::endl;
        #endif
            g_inh_device_to_host = t;
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
    
    }

    // Memory Management: track memory consumption
    long int size_in_bytes() {
        long int size_in_bytes = 0;
        // Parameters
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * Cm.capacity();	// Cm
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * Gm.capacity();	// Gm
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * bias.capacity();	// bias
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * tau.capacity();	// tau
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * To.capacity();	// To
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * m.capacity();	// m
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * tau_inh.capacity();	// tau_inh
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * Esyn.capacity();	// Esyn
        // Variables
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * v.capacity();	// v
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * T.capacity();	// T
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * g_inh.capacity();	// g_inh
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * r.capacity();	// r
        // RNGs
        
        return size_in_bytes;
    }

    // Memory Management: clear container
    void clear() {
        // Variables
        v.clear();
        v.shrink_to_fit();
        T.clear();
        T.shrink_to_fit();
        g_inh.clear();
        g_inh.shrink_to_fit();
        r.clear();
        r.shrink_to_fit();
        
        /* Free device variables */
        
        // Mean Firing Rate
        for (auto it = _spike_history.begin(); it != _spike_history.end(); it++) {
            while(!it->empty())
                it->pop();
        }
        _spike_history.clear();
        _spike_history.shrink_to_fit();
        // parameters
        cudaFree(gpu_Cm); 
        cudaFree(gpu_Gm); 
        cudaFree(gpu_bias); 
        cudaFree(gpu_tau); 
        cudaFree(gpu_To); 
        cudaFree(gpu_m); 
        cudaFree(gpu_tau_inh); 
        cudaFree(gpu_Esyn); 
        
        // variables
        cudaFree(gpu_v); 
        cudaFree(gpu_T); 
        cudaFree(gpu_g_inh); 
        cudaFree(gpu_r); 
        
        // delayed attributes
        
        // RNGs
        
        // targets
        
    #ifdef _DEBUG
        cudaError_t err_clear = cudaGetLastError();
        if ( err_clear != cudaSuccess )
            std::cout << "Pop0::clear() - cudaFree: " << cudaGetErrorString(err_clear) << std::endl;
    #endif

    }
};
