/*
 *  ANNarchy-version: 4.7.2.2
 */
#pragma once

#include "ANNarchy.h"
#include "CSRMatrixCUDA.hpp"




extern std::vector<std::mt19937> rng;
extern unsigned long long global_seed;

extern PopStruct2 pop2;
extern PopStruct2 pop2;


/////////////////////////////////////////////////////////////////////////////
// proj0: pop2 -> pop2 with target inh
/////////////////////////////////////////////////////////////////////////////
struct ProjStruct0 : CSRMatrixCUDA<int, int> {
    ProjStruct0() : CSRMatrixCUDA<int, int> ( 3, 3) {
    }

    // Launch configuration
    unsigned int _nb_blocks;
    unsigned int _threads_per_block;


    bool init_from_lil( std::vector<int> &row_indices,
                        std::vector< std::vector<int> > &column_indices,
                        std::vector< std::vector<double> > &values,
                        std::vector< std::vector<int> > &delays) {
        bool success = static_cast<CSRMatrixCUDA<int, int>*>(this)->init_matrix_from_lil(row_indices, column_indices);
        if (!success)
            return false;


        // Local variable w
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
        static_cast<CSRMatrixCUDA<int, int>*>(this)->print_data_representation();
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

    // Local parameter El
    std::vector< double > El;
    double* gpu_El;
    long int El_device_to_host;
    bool El_host_to_device;

    // Local parameter Eh
    std::vector< double > Eh;
    double* gpu_Eh;
    long int Eh_device_to_host;
    bool Eh_host_to_device;

    // Local variable w
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

        // Local parameter El
        El = init_matrix_variable<double>(0.0);
        gpu_El = init_matrix_variable_gpu<double>(El);
        El_host_to_device = true;
        El_device_to_host = t;

        // Local parameter Eh
        Eh = init_matrix_variable<double>(0.0);
        gpu_Eh = init_matrix_variable_gpu<double>(Eh);
        Eh_host_to_device = true;
        Eh_device_to_host = t;



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

        // Local parameter El
        if ( name.compare("El") == 0 ) {
            if ( El_device_to_host < t ) device_to_host();
            return get_matrix_variable_all<double>(El);
        }

        // Local parameter Eh
        if ( name.compare("Eh") == 0 ) {
            if ( Eh_device_to_host < t ) device_to_host();
            return get_matrix_variable_all<double>(Eh);
        }

        // Local variable w
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

        // Local parameter El
        if ( name.compare("El") == 0 ) {
            if ( El_device_to_host < t ) device_to_host();
            return get_matrix_variable_row<double>(El, rk_post);
        }

        // Local parameter Eh
        if ( name.compare("Eh") == 0 ) {
            if ( Eh_device_to_host < t ) device_to_host();
            return get_matrix_variable_row<double>(Eh, rk_post);
        }

        // Local variable w
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

        // Local parameter El
        if ( name.compare("El") == 0 ) {
            if ( El_device_to_host < t ) device_to_host();
            return get_matrix_variable<double>(El, rk_post, rk_pre);
        }

        // Local parameter Eh
        if ( name.compare("Eh") == 0 ) {
            if ( Eh_device_to_host < t ) device_to_host();
            return get_matrix_variable<double>(Eh, rk_post, rk_pre);
        }

        // Local variable w
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

        // Local parameter El
        if ( name.compare("El") == 0 ) {
            update_matrix_variable_all<double>(El, value);
            El_host_to_device = true;
            return;
        }

        // Local parameter Eh
        if ( name.compare("Eh") == 0 ) {
            update_matrix_variable_all<double>(Eh, value);
            Eh_host_to_device = true;
            return;
        }

        // Local variable w
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

        // Local parameter El
        if ( name.compare("El") == 0 ) {
            update_matrix_variable_row<double>(El, rk_post, value);
            El_host_to_device = true;
            return;
        }

        // Local parameter Eh
        if ( name.compare("Eh") == 0 ) {
            update_matrix_variable_row<double>(Eh, rk_post, value);
            Eh_host_to_device = true;
            return;
        }

        // Local variable w
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

        // Local parameter El
        if ( name.compare("El") == 0 ) {
            update_matrix_variable<double>(El, rk_post, rk_pre, value);
            El_host_to_device = true;
            return;
        }

        // Local parameter Eh
        if ( name.compare("Eh") == 0 ) {
            update_matrix_variable<double>(Eh, rk_post, rk_pre, value);
            Eh_host_to_device = true;
            return;
        }

        // Local variable w
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
        size_in_bytes += static_cast<CSRMatrixCUDA<int, int>*>(this)->size_in_bytes();

        // Local variable w
        size_in_bytes += sizeof(bool);
        size_in_bytes += sizeof(double*);
        size_in_bytes += sizeof(std::vector<double>);
        size_in_bytes += sizeof(double) * w.capacity();

        // Local parameter Gmax
        size_in_bytes += sizeof(bool);
        size_in_bytes += sizeof(double*);
        size_in_bytes += sizeof(std::vector<double>);
        size_in_bytes += sizeof(double) * Gmax.capacity();

        // Local parameter El
        size_in_bytes += sizeof(bool);
        size_in_bytes += sizeof(double*);
        size_in_bytes += sizeof(std::vector<double>);
        size_in_bytes += sizeof(double) * El.capacity();

        // Local parameter Eh
        size_in_bytes += sizeof(bool);
        size_in_bytes += sizeof(double*);
        size_in_bytes += sizeof(std::vector<double>);
        size_in_bytes += sizeof(double) * Eh.capacity();

        return size_in_bytes;
    }

    void clear() {
    #ifdef _DEBUG
        std::cout << "PopStruct0::clear()" << std::endl;
    #endif

        // Connectivity
        static_cast<CSRMatrixCUDA<int, int>*>(this)->clear();

        // w - host
        w.clear();
        w.shrink_to_fit();

        // w - device
        cudaFree(gpu_w);

        // Gmax - host
        Gmax.clear();
        Gmax.shrink_to_fit();

        // Gmax - device
        cudaFree(gpu_Gmax);

        // El - host
        El.clear();
        El.shrink_to_fit();

        // El - device
        cudaFree(gpu_El);

        // Eh - host
        Eh.clear();
        Eh.shrink_to_fit();

        // Eh - device
        cudaFree(gpu_Eh);

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

        // El: local
        if ( El_host_to_device )
        {
        #ifdef _DEBUG
            std::cout << "HtoD: El ( proj0 )" << std::endl;
        #endif
            cudaMemcpy( gpu_El, El.data(), num_non_zeros_ * sizeof( double ), cudaMemcpyHostToDevice);
            El_host_to_device = false;
        #ifdef _DEBUG
            cudaError_t err = cudaGetLastError();
            if ( err!= cudaSuccess )
                std::cout << "  error: " << cudaGetErrorString(err) << std::endl;
        #endif
        }

        // Eh: local
        if ( Eh_host_to_device )
        {
        #ifdef _DEBUG
            std::cout << "HtoD: Eh ( proj0 )" << std::endl;
        #endif
            cudaMemcpy( gpu_Eh, Eh.data(), num_non_zeros_ * sizeof( double ), cudaMemcpyHostToDevice);
            Eh_host_to_device = false;
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

        // El: local
        if ( El_device_to_host < t ) {
        #ifdef _DEBUG
            std::cout << "DtoH: El ( proj0 )" << std::endl;
        #endif
            cudaMemcpy( El.data(), gpu_El, num_non_zeros_ * sizeof( double ), cudaMemcpyDeviceToHost);
        #ifdef _DEBUG
            cudaError_t err_El = cudaGetLastError();
            if ( err_El != cudaSuccess )
                std::cout << "  error: " << cudaGetErrorString(err_El) << std::endl;
        #endif
            El_device_to_host = t;
        }

        // Eh: local
        if ( Eh_device_to_host < t ) {
        #ifdef _DEBUG
            std::cout << "DtoH: Eh ( proj0 )" << std::endl;
        #endif
            cudaMemcpy( Eh.data(), gpu_Eh, num_non_zeros_ * sizeof( double ), cudaMemcpyDeviceToHost);
        #ifdef _DEBUG
            cudaError_t err_Eh = cudaGetLastError();
            if ( err_Eh != cudaSuccess )
                std::cout << "  error: " << cudaGetErrorString(err_Eh) << std::endl;
        #endif
            Eh_device_to_host = t;
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
