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



    // Neuron specific parameters and variables

    // Local parameter Cm
    std::vector< double > Cm;

    // Local parameter Gm
    std::vector< double > Gm;

    // Local parameter bias
    std::vector< double > bias;

    // Local parameter Esyn
    std::vector< double > Esyn;

    // Local variable v
    std::vector< double > v;

    // Local variable r
    std::vector< double > r;

    // Local psp _sum_inh
    std::vector< double > _sum_inh;

    // Random numbers





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

        // Local parameter Esyn
        if ( name.compare("Esyn") == 0 ) {
            return Esyn;
        }

        // Local variable v
        if ( name.compare("v") == 0 ) {
            return v;
        }

        // Local variable r
        if ( name.compare("r") == 0 ) {
            return r;
        }

        // Local psp _sum_inh
        if ( name.compare("_sum_inh") == 0 ) {
            return _sum_inh;
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

        // Local parameter Esyn
        if ( name.compare("Esyn") == 0 ) {
            return Esyn[rk];
        }

        // Local variable v
        if ( name.compare("v") == 0 ) {
            return v[rk];
        }

        // Local variable r
        if ( name.compare("r") == 0 ) {
            return r[rk];
        }

        // Local psp _sum_inh
        if ( name.compare("_sum_inh") == 0 ) {
            return _sum_inh[rk];
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

        // Local variable r
        if ( name.compare("r") == 0 ) {
            r = value;
            return;
        }

        // Local psp _sum_inh
        if ( name.compare("_sum_inh") == 0 ) {
            _sum_inh = value;
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

        // Local variable r
        if ( name.compare("r") == 0 ) {
            r[rk] = value;
            return;
        }

        // Local psp _sum_inh
        if ( name.compare("_sum_inh") == 0 ) {
            _sum_inh[rk] = value;
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

        // Local parameter Esyn
        Esyn = std::vector<double>(size, 0.0);

        // Local variable v
        v = std::vector<double>(size, 0.0);

        // Local variable r
        r = std::vector<double>(size, 0.0);

        // Local psp _sum_inh
        _sum_inh = std::vector<double>(size, 0.0);






    }

    // Method called to reset the population
    void reset() {



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
        #ifdef _TRACE_SIMULATION_STEPS
            std::cout << "    PopStruct0::update()" << std::endl;
        #endif

            // Updating the local variables
            #pragma omp simd
            for(int i = 0; i < size; i++){

                // Cm * dv/dt = -Gm * v + bias + sum(inh)*(Esyn-v)
                double _v = (Esyn[i]*_sum_inh[i] - Gm[i]*v[i] - _sum_inh[i]*v[i] + bias[i])/Cm[i];

                // Cm * dv/dt = -Gm * v + bias + sum(inh)*(Esyn-v)
                v[i] += dt*_v ;


                // r = v
                r[i] = v[i];


            }
        } // active

    }

    void spike_gather() {

    }



    // Memory management: track the memory consumption
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

    // Memory management: destroy all the C++ data
    void clear() {
#ifdef _DEBUG
    std::cout << "PopStruct0::clear() - this = " << this << std::endl;
#endif
        // Variables
        v.clear();
        v.shrink_to_fit();
        r.clear();
        r.shrink_to_fit();

    }
};

