/*
 *  ANNarchy-version: 4.7.2.2
 */
#pragma once

#include "ANNarchy.h"
#include "LILInvMatrix.hpp"




extern PopStruct0 pop0;
extern PopStruct0 pop0;
extern double dt;
extern long int t;

extern std::vector<std::mt19937> rng;

/////////////////////////////////////////////////////////////////////////////
// proj0: pop0 -> pop0 with target inh
/////////////////////////////////////////////////////////////////////////////
struct ProjStruct0 : LILInvMatrix<int, int> {
    ProjStruct0() : LILInvMatrix<int, int>( 5000, 5000) {
    }


    bool init_from_lil( std::vector<int> &row_indices,
                        std::vector< std::vector<int> > &column_indices,
                        std::vector< std::vector<double> > &values,
                        std::vector< std::vector<int> > &delays) {
        bool success = static_cast<LILInvMatrix<int, int>*>(this)->init_matrix_from_lil(row_indices, column_indices);
        if (!success)
            return false;


        // Local parameter w
        w = init_matrix_variable<double>(static_cast<double>(0.0));
        update_matrix_variable_all<double>(w, values);


        // init other variables than 'w' or delay
        if (!init_attributes()){
            return false;
        }

    #ifdef _DEBUG_CONN
        static_cast<LILInvMatrix<int, int>*>(this)->print_data_representation();
    #endif
        return true;
    }





    // Transmission and plasticity flags
    bool _transmission, _plasticity, _update;
    int _update_period;
    long int _update_offset;





    // Local parameter Gmax
    std::vector< std::vector<double > > Gmax;

    // Local parameter Esyn
    std::vector< std::vector<double > > Esyn;

    // Local parameter w
    std::vector< std::vector<double > > w;




    // Method called to allocate/initialize the variables
    bool init_attributes() {

        // Local parameter Gmax
        Gmax = init_matrix_variable<double>(static_cast<double>(0.0));

        // Local parameter Esyn
        Esyn = init_matrix_variable<double>(static_cast<double>(0.0));




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
int nb_post; double sum;

        // Event-based summation
        if (_transmission && pop0._active){


            // Iterate over all incoming spikes (possibly delayed constantly)
            for(int _idx_j = 0; _idx_j < pop0.spiked.size(); _idx_j++){
                // Rank of the presynaptic neuron
                int rk_j = pop0.spiked[_idx_j];
                // Find the presynaptic neuron in the inverse connectivity matrix
                auto inv_post_ptr = inv_pre_rank.find(rk_j);
                if (inv_post_ptr == inv_pre_rank.end())
                    continue;
                // List of postsynaptic neurons receiving spikes from that neuron
                std::vector< std::pair<int, int> >& inv_post = inv_post_ptr->second;
                // Number of post neurons
                int nb_post = inv_post.size();

                // Iterate over connected post neurons
                for(int _idx_i = 0; _idx_i < nb_post; _idx_i++){
                    // Retrieve the correct indices
                    int i = inv_post[_idx_i].first;
                    int j = inv_post[_idx_i].second;

                    // Event-driven integration

                    // Update conductance

                    pop0.g_inh[post_rank[i]] +=  Gmax[i][j];

                    if (pop0.g_inh[post_rank[i]] > Gmax[i][j])
                        pop0.g_inh[post_rank[i]] = Gmax[i][j];

                    // Synaptic plasticity: pre-events

                }
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

        // Local parameter Esyn
        if ( name.compare("Esyn") == 0 ) {

            return get_matrix_variable_all<double>(Esyn);
        }

        // Local parameter w
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

        // Local parameter Esyn
        if ( name.compare("Esyn") == 0 ) {

            return get_matrix_variable_row<double>(Esyn, rk_post);
        }

        // Local parameter w
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

        // Local parameter Esyn
        if ( name.compare("Esyn") == 0 ) {

            return get_matrix_variable<double>(Esyn, rk_post, rk_pre);
        }

        // Local parameter w
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

        // Local parameter Esyn
        if ( name.compare("Esyn") == 0 ) {
            update_matrix_variable_all<double>(Esyn, value);

            return;
        }

        // Local parameter w
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

        // Local parameter Esyn
        if ( name.compare("Esyn") == 0 ) {
            update_matrix_variable_row<double>(Esyn, rk_post, value);

            return;
        }

        // Local parameter w
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

        // Local parameter Esyn
        if ( name.compare("Esyn") == 0 ) {
            update_matrix_variable<double>(Esyn, rk_post, rk_pre, value);

            return;
        }

        // Local parameter w
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
        size_in_bytes += static_cast<LILInvMatrix<int, int>*>(this)->size_in_bytes();

        // Local parameter Gmax
        size_in_bytes += sizeof(std::vector<std::vector<double>>);
        size_in_bytes += sizeof(std::vector<double>) * Gmax.capacity();
        for(auto it = Gmax.cbegin(); it != Gmax.cend(); it++)
            size_in_bytes += (it->capacity()) * sizeof(double);

        // Local parameter Esyn
        size_in_bytes += sizeof(std::vector<std::vector<double>>);
        size_in_bytes += sizeof(std::vector<double>) * Esyn.capacity();
        for(auto it = Esyn.cbegin(); it != Esyn.cend(); it++)
            size_in_bytes += (it->capacity()) * sizeof(double);

        // Local parameter w
        size_in_bytes += sizeof(std::vector<std::vector<double>>);
        size_in_bytes += sizeof(std::vector<double>) * w.capacity();
        for(auto it = w.cbegin(); it != w.cend(); it++)
            size_in_bytes += (it->capacity()) * sizeof(double);

        return size_in_bytes;
    }

    // Structural plasticity



    void clear() {
    #ifdef _DEBUG
        std::cout << "ProjStruct0::clear() - this = " << this << std::endl;
    #endif

        // Connectivity
        static_cast<LILInvMatrix<int, int>*>(this)->clear();

        // Gmax
        for (auto it = Gmax.begin(); it != Gmax.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        };
        Gmax.clear();
        Gmax.shrink_to_fit();

        // Esyn
        for (auto it = Esyn.begin(); it != Esyn.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        };
        Esyn.clear();
        Esyn.shrink_to_fit();

        // w
        for (auto it = w.begin(); it != w.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        };
        w.clear();
        w.shrink_to_fit();

    }
};

