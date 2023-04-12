/*
 *  ANNarchy-version: 4.7.2.2
 */
#pragma once

#include "ANNarchy.h"
#include "CSRCMatrixCUDA.hpp"




extern std::vector<std::mt19937> rng;
extern unsigned long long global_seed;

extern PopStruct2 pop2;
extern PopStruct2 pop2;


/////////////////////////////////////////////////////////////////////////////
// proj0: pop2 -> pop2 with target inh
/////////////////////////////////////////////////////////////////////////////
struct ProjStruct0 : CSRCMatrixCUDA<int, int> {
    ProjStruct0() : CSRCMatrixCUDA<int, int> ( 70, 70) {
    }

    // Launch configuration
    unsigned int _nb_blocks;
    unsigned int _threads_per_block;


    bool init_from_lil( std::vector<int> &row_indices,
                        std::vector< std::vector<int> > &column_indices,
                        std::vector< std::vector<double> > &values,
                        std::vector< std::vector<int> > &delays) {
        bool success = static_cast<CSRCMatrixCUDA<int, int>*>(this)->init_matrix_from_lil(row_indices, column_indices);
        if (!success)
            return false;


        // Local parameter w
        w = init_matrix_variable<double>(0.0);
        gpu_w = init_matrix_variable_gpu<double>(w);
        w_host_to_device = true;
        w_device_to_host = t;
        update_matrix_variable_all<double>(w, values);        
        w_host_to_device = true;


        // init other variables than 'w' or delay
        if (!init_attributes()){
            return false;
        }

    #ifdef _DEBUG_CONN
        static_cast<CSRCMatrixCUDA<int, int>*>(this)->print_data_representation();
    #endif
        return true;
    }





    // Transmission and plasticity flags
    bool _transmission, _plasticity, _update;
    int _update_period;
    long int _update_offset;





    // Local parameter Gmax
    std::vector< double > Gmax;
    double* gpu_Gmax;
    long int Gmax_device_to_host;
    bool Gmax_host_to_device;

    // Local parameter Esyn
    std::vector< double > Esyn;
    double* gpu_Esyn;
    long int Esyn_device_to_host;
    bool Esyn_host_to_device;

    // Local parameter w
    std::vector< double > w;
    double* gpu_w;
    long int w_device_to_host;
    bool w_host_to_device;


    // stream
    cudaStream_t stream;




    // Method called to allocate/initialize the variables
    bool init_attributes() {


        // Local parameter Gmax
        Gmax = init_matrix_variable<double>(0.0);
        gpu_Gmax = init_matrix_variable_gpu<double>(Gmax);
        Gmax_host_to_device = true;
        Gmax_device_to_host = t;

        // Local parameter Esyn
        Esyn = init_matrix_variable<double>(0.0);
        gpu_Esyn = init_matrix_variable_gpu<double>(Esyn);
        Esyn_host_to_device = true;
        Esyn_device_to_host = t;



        return true;
    }

    // Generate the default kernel launch configuration
    void default_launch_config() {

        _threads_per_block = 64;
        _nb_blocks = std::min<unsigned int>(nb_dendrites(), 65535);

    #ifdef _DEBUG
        std::cout << "Kernel configuration: " << _nb_blocks << ", " << _threads_per_block << std::endl;
    #endif

    }

    // Override the default kernel launch configuration
    void update_launch_config(int nb_blocks, int threads_per_block) {

        if (nb_blocks != -1) {
            _nb_blocks = static_cast<unsigned int>(nb_blocks);
            _threads_per_block = threads_per_block;
        }else{
            _threads_per_block = threads_per_block;
            _nb_blocks = std::min<unsigned int>(nb_dendrites(), 65535);
        }

    #ifdef _DEBUG
        std::cout << "Updated configuration: " << _nb_blocks << ", " << _threads_per_block << std::endl;
    #endif

    }

    // Method called to initialize the projection
    void init_projection() {
    #ifdef _DEBUG
        std::cout << "ProjStruct0::init_projection()" << std::endl;
    #endif

        _transmission = true;
        _update = true;
        _plasticity = true;
        _update_period = 1;
        _update_offset = 0L;

        default_launch_config();

        init_attributes();



    }

    // Additional access methods

    std::vector<std::vector<double>> get_local_attribute_all_double(std::string name) {
    #ifdef _DEBUG
        std::cout << "ProjStruct0::get_local_attribute_all_double(name = "<<name<<")" << std::endl;
    #endif

        // Local parameter Gmax
        if ( name.compare("Gmax") == 0 ) {
            if ( Gmax_device_to_host < t ) device_to_host();
            return get_matrix_variable_all<double>(Gmax);
        }

        // Local parameter Esyn
        if ( name.compare("Esyn") == 0 ) {
            if ( Esyn_device_to_host < t ) device_to_host();
            return get_matrix_variable_all<double>(Esyn);
        }

        // Local parameter w
        if ( name.compare("w") == 0 ) {
            if ( w_device_to_host < t ) device_to_host();
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
            if ( Gmax_device_to_host < t ) device_to_host();
            return get_matrix_variable_row<double>(Gmax, rk_post);
        }

        // Local parameter Esyn
        if ( name.compare("Esyn") == 0 ) {
            if ( Esyn_device_to_host < t ) device_to_host();
            return get_matrix_variable_row<double>(Esyn, rk_post);
        }

        // Local parameter w
        if ( name.compare("w") == 0 ) {
            if ( w_device_to_host < t ) device_to_host();
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
            if ( Gmax_device_to_host < t ) device_to_host();
            return get_matrix_variable<double>(Gmax, rk_post, rk_pre);
        }

        // Local parameter Esyn
        if ( name.compare("Esyn") == 0 ) {
            if ( Esyn_device_to_host < t ) device_to_host();
            return get_matrix_variable<double>(Esyn, rk_post, rk_pre);
        }

        // Local parameter w
        if ( name.compare("w") == 0 ) {
            if ( w_device_to_host < t ) device_to_host();
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
            Gmax_host_to_device = true;
            return;
        }

        // Local parameter Esyn
        if ( name.compare("Esyn") == 0 ) {
            update_matrix_variable_all<double>(Esyn, value);
            Esyn_host_to_device = true;
            return;
        }

        // Local parameter w
        if ( name.compare("w") == 0 ) {
            update_matrix_variable_all<double>(w, value);
            w_host_to_device = true;
            return;
        }

    }

    void set_local_attribute_row_double(std::string name, int rk_post, std::vector<double> value) {

        // Local parameter Gmax
        if ( name.compare("Gmax") == 0 ) {
            update_matrix_variable_row<double>(Gmax, rk_post, value);
            Gmax_host_to_device = true;
            return;
        }

        // Local parameter Esyn
        if ( name.compare("Esyn") == 0 ) {
            update_matrix_variable_row<double>(Esyn, rk_post, value);
            Esyn_host_to_device = true;
            return;
        }

        // Local parameter w
        if ( name.compare("w") == 0 ) {
            update_matrix_variable_row<double>(w, rk_post, value);
            w_host_to_device = true;
            return;
        }

    }

    void set_local_attribute_double(std::string name, int rk_post, int rk_pre, double value) {

        // Local parameter Gmax
        if ( name.compare("Gmax") == 0 ) {
            update_matrix_variable<double>(Gmax, rk_post, rk_pre, value);
            Gmax_host_to_device = true;
            return;
        }

        // Local parameter Esyn
        if ( name.compare("Esyn") == 0 ) {
            update_matrix_variable<double>(Esyn, rk_post, rk_pre, value);
            Esyn_host_to_device = true;
            return;
        }

        // Local parameter w
        if ( name.compare("w") == 0 ) {
            update_matrix_variable<double>(w, rk_post, rk_pre, value);
            w_host_to_device = true;
            return;
        }

    }



    // Memory management
    long int size_in_bytes() {
        long int size_in_bytes = 0;

        // connectivity
        size_in_bytes += static_cast<CSRCMatrixCUDA<int, int>*>(this)->size_in_bytes();

        // Local parameter Gmax
        size_in_bytes += sizeof(bool);
        size_in_bytes += sizeof(double*);
        size_in_bytes += sizeof(std::vector<double>);
        size_in_bytes += sizeof(double) * Gmax.capacity();

        // Local parameter Esyn
        size_in_bytes += sizeof(bool);
        size_in_bytes += sizeof(double*);
        size_in_bytes += sizeof(std::vector<double>);
        size_in_bytes += sizeof(double) * Esyn.capacity();

        // Local parameter w
        size_in_bytes += sizeof(bool);
        size_in_bytes += sizeof(double*);
        size_in_bytes += sizeof(std::vector<double>);
        size_in_bytes += sizeof(double) * w.capacity();

        return size_in_bytes;
    }

    void clear() {
    #ifdef _DEBUG
        std::cout << "PopStruct0::clear()" << std::endl;
    #endif

        // Connectivity
        static_cast<CSRCMatrixCUDA<int, int>*>(this)->clear();

        // Gmax - host
        Gmax.clear();
        Gmax.shrink_to_fit();

        // Gmax - device
        cudaFree(gpu_Gmax);

        // Esyn - host
        Esyn.clear();
        Esyn.shrink_to_fit();

        // Esyn - device
        cudaFree(gpu_Esyn);

        // w - host
        w.clear();
        w.shrink_to_fit();

        // w - device
        cudaFree(gpu_w);

    }

    // Memory transfers
    void host_to_device() {

        // Gmax: local
        if ( Gmax_host_to_device )
        {
        #ifdef _DEBUG
            std::cout << "HtoD: Gmax ( proj0 )" << std::endl;
        #endif
            cudaMemcpy( gpu_Gmax, Gmax.data(), num_non_zeros_ * sizeof( double ), cudaMemcpyHostToDevice);
            Gmax_host_to_device = false;
        #ifdef _DEBUG
            cudaError_t err = cudaGetLastError();
            if ( err!= cudaSuccess )
                std::cout << "  error: " << cudaGetErrorString(err) << std::endl;
        #endif
        }

        // Esyn: local
        if ( Esyn_host_to_device )
        {
        #ifdef _DEBUG
            std::cout << "HtoD: Esyn ( proj0 )" << std::endl;
        #endif
            cudaMemcpy( gpu_Esyn, Esyn.data(), num_non_zeros_ * sizeof( double ), cudaMemcpyHostToDevice);
            Esyn_host_to_device = false;
        #ifdef _DEBUG
            cudaError_t err = cudaGetLastError();
            if ( err!= cudaSuccess )
                std::cout << "  error: " << cudaGetErrorString(err) << std::endl;
        #endif
        }

        // w: local
        if ( w_host_to_device )
        {
        #ifdef _DEBUG
            std::cout << "HtoD: w ( proj0 )" << std::endl;
        #endif
            cudaMemcpy( gpu_w, w.data(), num_non_zeros_ * sizeof( double ), cudaMemcpyHostToDevice);
            w_host_to_device = false;
        #ifdef _DEBUG
            cudaError_t err = cudaGetLastError();
            if ( err!= cudaSuccess )
                std::cout << "  error: " << cudaGetErrorString(err) << std::endl;
        #endif
        }

    }

    void device_to_host() {

        // Gmax: local
        if ( Gmax_device_to_host < t ) {
        #ifdef _DEBUG
            std::cout << "DtoH: Gmax ( proj0 )" << std::endl;
        #endif
            cudaMemcpy( Gmax.data(), gpu_Gmax, num_non_zeros_ * sizeof( double ), cudaMemcpyDeviceToHost);
        #ifdef _DEBUG
            cudaError_t err_Gmax = cudaGetLastError();
            if ( err_Gmax != cudaSuccess )
                std::cout << "  error: " << cudaGetErrorString(err_Gmax) << std::endl;
        #endif
            Gmax_device_to_host = t;
        }

        // Esyn: local
        if ( Esyn_device_to_host < t ) {
        #ifdef _DEBUG
            std::cout << "DtoH: Esyn ( proj0 )" << std::endl;
        #endif
            cudaMemcpy( Esyn.data(), gpu_Esyn, num_non_zeros_ * sizeof( double ), cudaMemcpyDeviceToHost);
        #ifdef _DEBUG
            cudaError_t err_Esyn = cudaGetLastError();
            if ( err_Esyn != cudaSuccess )
                std::cout << "  error: " << cudaGetErrorString(err_Esyn) << std::endl;
        #endif
            Esyn_device_to_host = t;
        }

        // w: local
        if ( w_device_to_host < t ) {
        #ifdef _DEBUG
            std::cout << "DtoH: w ( proj0 )" << std::endl;
        #endif
            cudaMemcpy( w.data(), gpu_w, num_non_zeros_ * sizeof( double ), cudaMemcpyDeviceToHost);
        #ifdef _DEBUG
            cudaError_t err_w = cudaGetLastError();
            if ( err_w != cudaSuccess )
                std::cout << "  error: " << cudaGetErrorString(err_w) << std::endl;
        #endif
            w_device_to_host = t;
        }

    }
};
