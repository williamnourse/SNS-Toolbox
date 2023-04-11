/*
 *  ANNarchy-version: 4.7.2.2
 */
#pragma once

#include "ANNarchy.h"
#include "LILMatrix.hpp"




extern PopStruct0 pop0;
extern PopStruct0 pop0;
extern double dt;
extern long int t;

extern std::vector<std::mt19937> rng;

/////////////////////////////////////////////////////////////////////////////
// proj0: pop0 -> pop0 with target inh
/////////////////////////////////////////////////////////////////////////////
struct ProjStruct0 : LILMatrix<int, int> {
    ProjStruct0() : LILMatrix<int, int>( 16, 16) {
    }


    bool init_from_lil( std::vector<int> &row_indices,
                        std::vector< std::vector<int> > &column_indices,
                        std::vector< std::vector<double> > &values,
                        std::vector< std::vector<int> > &delays) {
        bool success = static_cast<LILMatrix<int, int>*>(this)->init_matrix_from_lil(row_indices, column_indices);
        if (!success)
            return false;


        // Local variable w
        w = init_matrix_variable<double>(static_cast<double>(0.0));
        update_matrix_variable_all<double>(w, values);


        // init other variables than 'w' or delay
        if (!init_attributes()){
            return false;
        }

    #ifdef _DEBUG_CONN
        static_cast<LILMatrix<int, int>*>(this)->print_data_representation();
    #endif
        return true;
    }





    // Transmission and plasticity flags
    bool _transmission, _plasticity, _update;
    int _update_period;
    long int _update_offset;





    // Local parameter Gmax
    std::vector< std::vector<double > > Gmax;

    // Local parameter El
    std::vector< std::vector<double > > El;

    // Local parameter Eh
    std::vector< std::vector<double > > Eh;

    // Local variable w
    std::vector< std::vector<double > > w;




    // Method called to allocate/initialize the variables
    bool init_attributes() {

        // Local parameter Gmax
        Gmax = init_matrix_variable<double>(static_cast<double>(0.0));

        // Local parameter El
        El = init_matrix_variable<double>(static_cast<double>(0.0));

        // Local parameter Eh
        Eh = init_matrix_variable<double>(static_cast<double>(0.0));




        return true;
    }

    // Method called to initialize the projection
    void init_projection() {
    #ifdef _DEBUG
        std::cout << "ProjStruct0::init_projection() - this = " << this << std::endl;
    #endif

        _transmission = true;
        _update = true;
        _plasticity = true;
        _update_period = 1;
        _update_offset = 0L;

        init_attributes();



    }

    // Spiking networks: reset the ring buffer when non-uniform
    void reset_ring_buffer() {

    }

    // Spiking networks: update maximum delay when non-uniform
    void update_max_delay(int d){

    }

    // Computes the weighted sum of inputs or updates the conductances
    void compute_psp() {
    #ifdef _TRACE_SIMULATION_STEPS
        std::cout << "    ProjStruct0::compute_psp()" << std::endl;
    #endif
        double sum;

        if (_transmission && pop0._active) {



            for (int i = 0; i < post_rank.size(); i++) {

                sum = 0.0;
                for (int j = 0; j < pre_rank[i].size(); j++) {
                    sum += w[i][j] ;
                }
                pop0._sum_inh[post_rank[i]] += sum;
            }

        } // active

    }

    // Draws random numbers
    void update_rng() {

    }

    // Updates synaptic variables
    void update_synapse() {
    #ifdef _TRACE_SIMULATION_STEPS
        std::cout << "    ProjStruct0::update_synapse()" << std::endl;
    #endif

        int rk_post, rk_pre;
        double _dt = dt * _update_period;

        // Check periodicity
        if(_transmission && _update && pop0._active && ( (t - _update_offset)%_update_period == 0L) ){
            // Global variables


            // Semiglobal/Local variables
            for (int i = 0; i < post_rank.size(); i++) {
                rk_post = post_rank[i]; // Get postsynaptic rank

                // Semi-global variables


                // Local variables
                for (int j = 0; j < pre_rank[i].size(); j++) {
                    rk_pre = pre_rank[i][j]; // Get presynaptic rank

                    // w = clip(Gmax * (pre.r-El)/(Eh-El), 0.0, Gmax)
                    if(_plasticity){
                    w[i][j] = clip(Gmax[i][j]*(-El[i][j] + pop0.r[rk_pre])/(Eh[i][j] - El[i][j]), 0.0, Gmax[i][j]);

                    }

                }
            }
        }

    }

    // Post-synaptic events
    void post_event() {

    }

    // Variable/Parameter access methods

    std::vector<std::vector<double>> get_local_attribute_all_double(std::string name) {
    #ifdef _DEBUG
        std::cout << "ProjStruct0::get_local_attribute_all_double(name = "<<name<<")" << std::endl;
    #endif

        // Local parameter Gmax
        if ( name.compare("Gmax") == 0 ) {

            return get_matrix_variable_all<double>(Gmax);
        }

        // Local parameter El
        if ( name.compare("El") == 0 ) {

            return get_matrix_variable_all<double>(El);
        }

        // Local parameter Eh
        if ( name.compare("Eh") == 0 ) {

            return get_matrix_variable_all<double>(Eh);
        }

        // Local variable w
        if ( name.compare("w") == 0 ) {

            return get_matrix_variable_all<double>(w);
        }


        // should not happen
        std::cerr << "ProjStruct0::get_local_attribute_all_double: " << name << " not found" << std::endl;
        return std::vector<std::vector<double>>();
    }

    std::vector<double> get_local_attribute_row_double(std::string name, int rk_post) {
    #ifdef _DEBUG
        std::cout << "ProjStruct0::get_local_attribute_row_double(name = "<<name<<", rk_post = "<<rk_post<<")" << std::endl;
    #endif

        // Local parameter Gmax
        if ( name.compare("Gmax") == 0 ) {

            return get_matrix_variable_row<double>(Gmax, rk_post);
        }

        // Local parameter El
        if ( name.compare("El") == 0 ) {

            return get_matrix_variable_row<double>(El, rk_post);
        }

        // Local parameter Eh
        if ( name.compare("Eh") == 0 ) {

            return get_matrix_variable_row<double>(Eh, rk_post);
        }

        // Local variable w
        if ( name.compare("w") == 0 ) {

            return get_matrix_variable_row<double>(w, rk_post);
        }


        // should not happen
        std::cerr << "ProjStruct0::get_local_attribute_row_double: " << name << " not found" << std::endl;
        return std::vector<double>();
    }

    double get_local_attribute_double(std::string name, int rk_post, int rk_pre) {
    #ifdef _DEBUG
        std::cout << "ProjStruct0::get_local_attribute_row_double(name = "<<name<<", rk_post = "<<rk_post<<", rk_pre = "<<rk_pre<<")" << std::endl;
    #endif

        // Local parameter Gmax
        if ( name.compare("Gmax") == 0 ) {

            return get_matrix_variable<double>(Gmax, rk_post, rk_pre);
        }

        // Local parameter El
        if ( name.compare("El") == 0 ) {

            return get_matrix_variable<double>(El, rk_post, rk_pre);
        }

        // Local parameter Eh
        if ( name.compare("Eh") == 0 ) {

            return get_matrix_variable<double>(Eh, rk_post, rk_pre);
        }

        // Local variable w
        if ( name.compare("w") == 0 ) {

            return get_matrix_variable<double>(w, rk_post, rk_pre);
        }


        // should not happen
        std::cerr << "ProjStruct0::get_local_attribute: " << name << " not found" << std::endl;
        return 0.0;
    }

    void set_local_attribute_all_double(std::string name, std::vector<std::vector<double>> value) {

        // Local parameter Gmax
        if ( name.compare("Gmax") == 0 ) {
            update_matrix_variable_all<double>(Gmax, value);

            return;
        }

        // Local parameter El
        if ( name.compare("El") == 0 ) {
            update_matrix_variable_all<double>(El, value);

            return;
        }

        // Local parameter Eh
        if ( name.compare("Eh") == 0 ) {
            update_matrix_variable_all<double>(Eh, value);

            return;
        }

        // Local variable w
        if ( name.compare("w") == 0 ) {
            update_matrix_variable_all<double>(w, value);

            return;
        }

    }

    void set_local_attribute_row_double(std::string name, int rk_post, std::vector<double> value) {

        // Local parameter Gmax
        if ( name.compare("Gmax") == 0 ) {
            update_matrix_variable_row<double>(Gmax, rk_post, value);

            return;
        }

        // Local parameter El
        if ( name.compare("El") == 0 ) {
            update_matrix_variable_row<double>(El, rk_post, value);

            return;
        }

        // Local parameter Eh
        if ( name.compare("Eh") == 0 ) {
            update_matrix_variable_row<double>(Eh, rk_post, value);

            return;
        }

        // Local variable w
        if ( name.compare("w") == 0 ) {
            update_matrix_variable_row<double>(w, rk_post, value);

            return;
        }

    }

    void set_local_attribute_double(std::string name, int rk_post, int rk_pre, double value) {

        // Local parameter Gmax
        if ( name.compare("Gmax") == 0 ) {
            update_matrix_variable<double>(Gmax, rk_post, rk_pre, value);

            return;
        }

        // Local parameter El
        if ( name.compare("El") == 0 ) {
            update_matrix_variable<double>(El, rk_post, rk_pre, value);

            return;
        }

        // Local parameter Eh
        if ( name.compare("Eh") == 0 ) {
            update_matrix_variable<double>(Eh, rk_post, rk_pre, value);

            return;
        }

        // Local variable w
        if ( name.compare("w") == 0 ) {
            update_matrix_variable<double>(w, rk_post, rk_pre, value);

            return;
        }

    }


    // Access additional


    // Memory management
    long int size_in_bytes() {
        long int size_in_bytes = 0;

        // connectivity
        size_in_bytes += static_cast<LILMatrix<int, int>*>(this)->size_in_bytes();

        // Local variable w
        size_in_bytes += sizeof(std::vector<std::vector<double>>);
        size_in_bytes += sizeof(std::vector<double>) * w.capacity();
        for(auto it = w.cbegin(); it != w.cend(); it++)
            size_in_bytes += (it->capacity()) * sizeof(double);

        // Local parameter Gmax
        size_in_bytes += sizeof(std::vector<std::vector<double>>);
        size_in_bytes += sizeof(std::vector<double>) * Gmax.capacity();
        for(auto it = Gmax.cbegin(); it != Gmax.cend(); it++)
            size_in_bytes += (it->capacity()) * sizeof(double);

        // Local parameter El
        size_in_bytes += sizeof(std::vector<std::vector<double>>);
        size_in_bytes += sizeof(std::vector<double>) * El.capacity();
        for(auto it = El.cbegin(); it != El.cend(); it++)
            size_in_bytes += (it->capacity()) * sizeof(double);

        // Local parameter Eh
        size_in_bytes += sizeof(std::vector<std::vector<double>>);
        size_in_bytes += sizeof(std::vector<double>) * Eh.capacity();
        for(auto it = Eh.cbegin(); it != Eh.cend(); it++)
            size_in_bytes += (it->capacity()) * sizeof(double);

        return size_in_bytes;
    }

    // Structural plasticity



    void clear() {
    #ifdef _DEBUG
        std::cout << "ProjStruct0::clear() - this = " << this << std::endl;
    #endif

        // Connectivity
        static_cast<LILMatrix<int, int>*>(this)->clear();

        // w
        for (auto it = w.begin(); it != w.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        };
        w.clear();
        w.shrink_to_fit();

        // Gmax
        for (auto it = Gmax.begin(); it != Gmax.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        };
        Gmax.clear();
        Gmax.shrink_to_fit();

        // El
        for (auto it = El.begin(); it != El.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        };
        El.clear();
        El.shrink_to_fit();

        // Eh
        for (auto it = Eh.begin(); it != Eh.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        };
        Eh.clear();
        Eh.shrink_to_fit();

    }
};

