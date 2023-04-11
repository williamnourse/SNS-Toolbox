/*
 *  ANNarchy-version: 4.7.2.2
 */
#pragma once

#include "ANNarchy.h"
#include <random>



extern double dt;
extern long int t;
extern std::vector<std::mt19937> rng;


///////////////////////////////////////////////////////////////
// Main Structure for the population of id 0 (pop0)
///////////////////////////////////////////////////////////////
struct PopStruct0{

    int size; // Number of neurons
    bool _active; // Allows to shut down the whole population
    int max_delay; // Maximum number of steps to store for delayed synaptic transmission

    // Access functions used by cython wrapper
    int get_size() { return size; }
    void set_size(int s) { size  = s; }
    int get_max_delay() { return max_delay; }
    void set_max_delay(int d) { max_delay  = d; }
    bool is_active() { return _active; }
    void set_active(bool val) { _active = val; }



    // Structures for managing spikes
    std::vector<long int> last_spike;
    std::vector<int> spiked;

    // Neuron specific parameters and variables

    // Local parameter Cm
    std::vector< double > Cm;

    // Local parameter Gm
    std::vector< double > Gm;

    // Local parameter bias
    std::vector< double > bias;

    // Local parameter tau
    std::vector< double > tau;

    // Local parameter To
    std::vector< double > To;

    // Local parameter m
    std::vector< double > m;

    // Local parameter tau_inh
    std::vector< double > tau_inh;

    // Local parameter Esyn
    std::vector< double > Esyn;

    // Local variable v
    std::vector< double > v;

    // Local variable T
    std::vector< double > T;

    // Local variable g_inh
    std::vector< double > g_inh;

    // Local variable r
    std::vector< double > r;

    // Random numbers



    // Mean Firing rate
    std::vector< std::queue<long int> > _spike_history;
    long int _mean_fr_window;
    double _mean_fr_rate;
    void compute_firing_rate(double window){
        if(window>0.0){
            _mean_fr_window = int(window/dt);
            _mean_fr_rate = 1000./window;
            if (_spike_history.empty())
                _spike_history = std::vector< std::queue<long int> >(size, std::queue<long int>());
        }
    };


    // Access methods to the parameters and variables

    std::vector<double> get_local_attribute_all_double(std::string name) {

        // Local parameter Cm
        if ( name.compare("Cm") == 0 ) {
            return Cm;
        }

        // Local parameter Gm
        if ( name.compare("Gm") == 0 ) {
            return Gm;
        }

        // Local parameter bias
        if ( name.compare("bias") == 0 ) {
            return bias;
        }

        // Local parameter tau
        if ( name.compare("tau") == 0 ) {
            return tau;
        }

        // Local parameter To
        if ( name.compare("To") == 0 ) {
            return To;
        }

        // Local parameter m
        if ( name.compare("m") == 0 ) {
            return m;
        }

        // Local parameter tau_inh
        if ( name.compare("tau_inh") == 0 ) {
            return tau_inh;
        }

        // Local parameter Esyn
        if ( name.compare("Esyn") == 0 ) {
            return Esyn;
        }

        // Local variable v
        if ( name.compare("v") == 0 ) {
            return v;
        }

        // Local variable T
        if ( name.compare("T") == 0 ) {
            return T;
        }

        // Local variable g_inh
        if ( name.compare("g_inh") == 0 ) {
            return g_inh;
        }

        // Local variable r
        if ( name.compare("r") == 0 ) {
            return r;
        }


        // should not happen
        std::cerr << "PopStruct0::get_local_attribute_all_double: " << name << " not found" << std::endl;
        return std::vector<double>();
    }

    double get_local_attribute_double(std::string name, int rk) {
        assert( (rk < size) );

        // Local parameter Cm
        if ( name.compare("Cm") == 0 ) {
            return Cm[rk];
        }

        // Local parameter Gm
        if ( name.compare("Gm") == 0 ) {
            return Gm[rk];
        }

        // Local parameter bias
        if ( name.compare("bias") == 0 ) {
            return bias[rk];
        }

        // Local parameter tau
        if ( name.compare("tau") == 0 ) {
            return tau[rk];
        }

        // Local parameter To
        if ( name.compare("To") == 0 ) {
            return To[rk];
        }

        // Local parameter m
        if ( name.compare("m") == 0 ) {
            return m[rk];
        }

        // Local parameter tau_inh
        if ( name.compare("tau_inh") == 0 ) {
            return tau_inh[rk];
        }

        // Local parameter Esyn
        if ( name.compare("Esyn") == 0 ) {
            return Esyn[rk];
        }

        // Local variable v
        if ( name.compare("v") == 0 ) {
            return v[rk];
        }

        // Local variable T
        if ( name.compare("T") == 0 ) {
            return T[rk];
        }

        // Local variable g_inh
        if ( name.compare("g_inh") == 0 ) {
            return g_inh[rk];
        }

        // Local variable r
        if ( name.compare("r") == 0 ) {
            return r[rk];
        }


        // should not happen
        std::cerr << "PopStruct0::get_local_attribute_double: " << name << " not found" << std::endl;
        return static_cast<double>(0.0);
    }

    void set_local_attribute_all_double(std::string name, std::vector<double> value) {
        assert( (value.size() == size) );

        // Local parameter Cm
        if ( name.compare("Cm") == 0 ) {
            Cm = value;
            return;
        }

        // Local parameter Gm
        if ( name.compare("Gm") == 0 ) {
            Gm = value;
            return;
        }

        // Local parameter bias
        if ( name.compare("bias") == 0 ) {
            bias = value;
            return;
        }

        // Local parameter tau
        if ( name.compare("tau") == 0 ) {
            tau = value;
            return;
        }

        // Local parameter To
        if ( name.compare("To") == 0 ) {
            To = value;
            return;
        }

        // Local parameter m
        if ( name.compare("m") == 0 ) {
            m = value;
            return;
        }

        // Local parameter tau_inh
        if ( name.compare("tau_inh") == 0 ) {
            tau_inh = value;
            return;
        }

        // Local parameter Esyn
        if ( name.compare("Esyn") == 0 ) {
            Esyn = value;
            return;
        }

        // Local variable v
        if ( name.compare("v") == 0 ) {
            v = value;
            return;
        }

        // Local variable T
        if ( name.compare("T") == 0 ) {
            T = value;
            return;
        }

        // Local variable g_inh
        if ( name.compare("g_inh") == 0 ) {
            g_inh = value;
            return;
        }

        // Local variable r
        if ( name.compare("r") == 0 ) {
            r = value;
            return;
        }


        // should not happen
        std::cerr << "PopStruct0::set_local_attribute_all_double: " << name << " not found" << std::endl;
    }

    void set_local_attribute_double(std::string name, int rk, double value) {
        assert( (rk < size) );

        // Local parameter Cm
        if ( name.compare("Cm") == 0 ) {
            Cm[rk] = value;
            return;
        }

        // Local parameter Gm
        if ( name.compare("Gm") == 0 ) {
            Gm[rk] = value;
            return;
        }

        // Local parameter bias
        if ( name.compare("bias") == 0 ) {
            bias[rk] = value;
            return;
        }

        // Local parameter tau
        if ( name.compare("tau") == 0 ) {
            tau[rk] = value;
            return;
        }

        // Local parameter To
        if ( name.compare("To") == 0 ) {
            To[rk] = value;
            return;
        }

        // Local parameter m
        if ( name.compare("m") == 0 ) {
            m[rk] = value;
            return;
        }

        // Local parameter tau_inh
        if ( name.compare("tau_inh") == 0 ) {
            tau_inh[rk] = value;
            return;
        }

        // Local parameter Esyn
        if ( name.compare("Esyn") == 0 ) {
            Esyn[rk] = value;
            return;
        }

        // Local variable v
        if ( name.compare("v") == 0 ) {
            v[rk] = value;
            return;
        }

        // Local variable T
        if ( name.compare("T") == 0 ) {
            T[rk] = value;
            return;
        }

        // Local variable g_inh
        if ( name.compare("g_inh") == 0 ) {
            g_inh[rk] = value;
            return;
        }

        // Local variable r
        if ( name.compare("r") == 0 ) {
            r[rk] = value;
            return;
        }


        // should not happen
        std::cerr << "PopStruct0::set_local_attribute_double: " << name << " not found" << std::endl;
    }



    // Method called to initialize the data structures
    void init_population() {
    #ifdef _DEBUG
        std::cout << "PopStruct0::init_population(size="<<this->size<<") - this = " << this << std::endl;
    #endif
        _active = true;

        // Local parameter Cm
        Cm = std::vector<double>(size, 0.0);

        // Local parameter Gm
        Gm = std::vector<double>(size, 0.0);

        // Local parameter bias
        bias = std::vector<double>(size, 0.0);

        // Local parameter tau
        tau = std::vector<double>(size, 0.0);

        // Local parameter To
        To = std::vector<double>(size, 0.0);

        // Local parameter m
        m = std::vector<double>(size, 0.0);

        // Local parameter tau_inh
        tau_inh = std::vector<double>(size, 0.0);

        // Local parameter Esyn
        Esyn = std::vector<double>(size, 0.0);

        // Local variable v
        v = std::vector<double>(size, 0.0);

        // Local variable T
        T = std::vector<double>(size, 0.0);

        // Local variable g_inh
        g_inh = std::vector<double>(size, 0.0);

        // Local variable r
        r = std::vector<double>(size, 0.0);


        // Spiking variables
        spiked = std::vector<int>();
        last_spike = std::vector<long int>(size, -10000L);



        // Mean Firing Rate
        _spike_history = std::vector< std::queue<long int> >();
        _mean_fr_window = 0;
        _mean_fr_rate = 1.0;


    }

    // Method called to reset the population
    void reset() {

        // Spiking variables
        spiked.clear();
        spiked.shrink_to_fit();
        std::fill(last_spike.begin(), last_spike.end(), -10000L);

        // Mean Firing Rate
        for (auto it = _spike_history.begin(); it != _spike_history.end(); it++) {
            if (!it->empty()) {
                auto empty_queue = std::queue<long int>();
                it->swap(empty_queue);
            }
        }



    }

    // Method to draw new random numbers
    void update_rng() {
#ifdef _TRACE_SIMULATION_STEPS
    std::cout << "    PopStruct0::update_rng()" << std::endl;
#endif

    }

    // Method to update global operations on the population (min/max/mean...)
    void update_global_ops() {

    }

    // Method to enqueue output variables in case outgoing projections have non-zero delay
    void update_delay() {

    }

    // Method to dynamically change the size of the queue for delayed variables
    void update_max_delay(int value) {

    }

    // Main method to update neural variables
    void update() {

        if( _active ) {



            // Updating local variables
            #pragma omp simd
            for(int i = 0; i < size; i++){

                // Cm * dv/dt = -Gm * v + bias + g_inh * (Esyn-v)
                double _v = (Esyn[i]*g_inh[i] - Gm[i]*v[i] + bias[i] - g_inh[i]*v[i])/Cm[i];

                // tau * dT/dt = -T + To + m * v
                double _T = (-T[i] + To[i] + m[i]*v[i])/tau[i];

                // tau_inh * dg_inh/dt = -g_inh
                double _g_inh = -g_inh[i]/tau_inh[i];

                // Cm * dv/dt = -Gm * v + bias + g_inh * (Esyn-v)
                v[i] += dt*_v ;


                // tau * dT/dt = -T + To + m * v
                T[i] += dt*_T ;


                // tau_inh * dg_inh/dt = -g_inh
                g_inh[i] += dt*_g_inh ;


            }
        } // active

    }

    void spike_gather() {

        if( _active ) {
            spiked.clear();

            for (int i = 0; i < size; i++) {


                // Spike emission
                if(v[i] > T[i]){ // Condition is met
                    // Reset variables

                    v[i] = 0;

                    // Store the spike
                    spiked.push_back(i);
                    last_spike[i] = t;

                    // Refractory period


                    // Store the event for the mean firing rate
                    if (_mean_fr_window > 0)
                        _spike_history[i].push(t);

                }

            }

            // Update mean firing rate
            if (_mean_fr_window > 0) {
                for (int i = 0; i < size; i++) {
                    while((_spike_history[i].size() != 0)&&(_spike_history[i].front() <= t - _mean_fr_window)){
                        _spike_history[i].pop(); // Suppress spikes outside the window
                    }
                    r[i] = _mean_fr_rate * double(_spike_history[i].size());
                }
            }
        } // active

    }



    // Memory management: track the memory consumption
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

    // Memory management: destroy all the C++ data
    void clear() {
#ifdef _DEBUG
    std::cout << "PopStruct0::clear() - this = " << this << std::endl;
#endif
        // Variables
        v.clear();
        v.shrink_to_fit();
        T.clear();
        T.shrink_to_fit();
        g_inh.clear();
        g_inh.shrink_to_fit();
        r.clear();
        r.shrink_to_fit();

        // Mean Firing Rate
        for (auto it = _spike_history.begin(); it != _spike_history.end(); it++) {
            while(!it->empty())
                it->pop();
        }
        _spike_history.clear();
        _spike_history.shrink_to_fit();

    }
};

