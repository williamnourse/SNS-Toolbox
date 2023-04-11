#ifndef __ANNARCHY_H__
#define __ANNARCHY_H__

#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include <deque>
#include <queue>
#include <random>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cstdlib>
#include <stdlib.h>
#include <string.h>
#include <cassert>

#include <cuda_runtime_api.h>
#include <curand_kernel.h>

/*
 * Built-in functions (host side)
 */

#define positive(x) (x>0.0? x : 0.0)
#define negative(x) (x<0.0? x : 0.0)
#define clip(x, a, b) (x<a? a : (x>b? b :x))
#define modulo(a, b) long(a) % long(b)
#define ite(a, b, c) (a?b:c)


/*
 * Custom constants
 *
 */


/*
 * Custom functions
 * (available on host-side and interfaced for cython)
 */


/*
 * Structures for the populations
 *
 */
#include "pop0.hpp"
#include "pop1.hpp"
#include "pop2.hpp"
#include "pop3.hpp"

/*
 * Structures for the projections
 *
 */
#include "proj0.hpp"


/*
 * Declaration of the populations
 *
 */
extern PopStruct0 pop0;
extern PopStruct1 pop1;
extern PopStruct2 pop2;
extern PopStruct3 pop3;


/*
 * Declaration of the projections
 *
 */
extern ProjStruct0 proj0;


/*
 * Recorders
 *
 */
#include "Recorder.h"

extern std::vector<Monitor*> recorders;
int addRecorder(Monitor* recorder);
Monitor* getRecorder(int id);
void removeRecorder(Monitor* recorder);

/*
 * Simulation methods
 */
void run(int nbSteps);

int run_until(int steps, std::vector<int> populations, bool or_and);

void step();

/*
 *  Initialization
 */
void initialize(const double _dt) ;

inline void setDevice(const int device_id) {
#ifdef _DEBUG
    std::cout << "Setting device " << device_id << " as compute device ..." << std::endl;
#endif
    cudaError_t err = cudaSetDevice(device_id);
    if ( err != cudaSuccess )
        std::cerr << "Set device " << device_id << ": " << cudaGetErrorString(err) << std::endl;
}

/*
 * Time export
 */
long int getTime();
void setTime(const long int t_);
double getDt();
void setDt(const double dt_);

/*
 * Seed for the RNG (host-side!)
 */
void setSeed(const long int seed, const int num_sources, const bool use_seed_seq);

#endif
