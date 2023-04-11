
#include "ANNarchy.h"



/*
 * Internal data
 *
 */
double dt;
long int t;
std::vector<std::mt19937> rng;

// Custom constants


// Populations
PopStruct0 pop0;


// Projections
ProjStruct0 proj0;


// Global operations


/*
 * Recorders
 */
std::vector<Monitor*> recorders;
int addRecorder(Monitor* recorder){
    int found = -1;

    for (unsigned int i=0; i<recorders.size(); i++) {
        if (recorders[i] == nullptr) {
            found = i;
            break;
        }
    }

    if (found != -1) {
        // fill a previously cleared slot
        recorders[found] = recorder;
        return found;
    } else {
        recorders.push_back(recorder);
        return recorders.size() - 1;
    }
}
Monitor* getRecorder(int id) {
    if (id < recorders.size())
        return recorders[id];
    else
        return nullptr;
}
void removeRecorder(Monitor* recorder){
    for (unsigned int i=0; i<recorders.size(); i++){
        if(recorders[i] == recorder){
            recorders[i] = nullptr;
            break;
        }
    }
}

/*
 *  Simulation methods
 */
// Simulate a single step
void singleStep()
{


    ////////////////////////////////
    // Presynaptic events
    ////////////////////////////////


#ifdef _TRACE_SIMULATION_STEPS
    std::cout << "Update psp/conductances ..." << std::endl;
#endif
	proj0.compute_psp();



    ////////////////////////////////
    // Recording target variables
    ////////////////////////////////
#ifdef _TRACE_SIMULATION_STEPS
    std::cout << "Record psp/conductances ..." << std::endl;
#endif
    for (unsigned int i=0; i < recorders.size(); i++) {
        if (recorders[i])
            recorders[i]->record_targets();
    }

    ////////////////////////////////
    // Update random distributions
    ////////////////////////////////
#ifdef _TRACE_SIMULATION_STEPS
    std::cout << "Draw required random numbers ..." << std::endl;
#endif




    ////////////////////////////////
    // Update neural variables
    ////////////////////////////////
#ifdef _TRACE_SIMULATION_STEPS
    std::cout << "Evaluate neural ODEs ..." << std::endl;
#endif

	pop0.update();
	pop0.spike_gather();



    ////////////////////////////////
    // Delay outputs
    ////////////////////////////////
#ifdef _TRACE_SIMULATION_STEPS
    std::cout << "Update delay queues ..." << std::endl;
#endif


    ////////////////////////////////
    // Global operations (min/max/mean)
    ////////////////////////////////
#ifdef _TRACE_SIMULATION_STEPS
    std::cout << "Update global operations ..." << std::endl;
#endif




    ////////////////////////////////
    // Update synaptic variables
    ////////////////////////////////
#ifdef _TRACE_SIMULATION_STEPS
    std::cout << "Evaluate synaptic ODEs ..." << std::endl;
#endif




    ////////////////////////////////
    // Postsynaptic events
    ////////////////////////////////




    ////////////////////////////////
    // Structural plasticity
    ////////////////////////////////


    ////////////////////////////////
    // Recording neural / synaptic variables
    ////////////////////////////////

    for (unsigned int i=0; i < recorders.size(); i++){
        if (recorders[i])
            recorders[i]->record();
    }


    ////////////////////////////////
    // Increase internal time
    ////////////////////////////////
    t++;


}

// Simulate the network for the given number of steps,
// called from python
void run(const int nbSteps) {
#ifdef _TRACE_SIMULATION_STEPS
    std::cout << "Perform simulation for " << nbSteps << " steps." << std::endl;
#endif

    for(int i=0; i<nbSteps; i++) {
        singleStep();
    }

}

// Simulate the network for a single steps,
// called from python
void step() {

    singleStep();

}

int run_until(const int steps, std::vector<int> populations, bool or_and)
{


    run(steps);
    return steps;


}

/*
 *  Initialization methods
 */
// Initialize the internal data and the random numbers generator
void initialize(const double _dt) {


    // Internal variables
    dt = _dt;
    t = static_cast<long int>(0);

    // Populations
    // Initialize populations
    pop0.init_population();


    // Projections
    // Initialize projections
    proj0.init_projection();


    // Custom constants


}

// Change the seed of the RNG
void setSeed(const long int seed, const int num_sources, const bool use_seed_seq) {
    if (num_sources > 1)
        std::cerr << "WARNING - ANNarchy::setSeed(): num_sources should be 1 for single thread code." << std::endl;

    rng.clear();

    rng.push_back(std::mt19937(seed));

    rng.shrink_to_fit();
}

/*
 * Access to time and dt
 */
long int getTime() {return t;}
void setTime(const long int t_) { t=t_;}
double getDt() { return dt;}
void setDt(const double dt_) { dt=dt_;}

/*
 * Number of threads
 *
*/
void setNumberThreads(const int threads, const std::vector<int> core_list)
{
    if (threads > 1) {
        std::cerr << "WARNING: a call of setNumberThreads() is without effect on single thread simulation code." << std::endl;
    }

    if (core_list.size()>1) {
        std::cerr << "The provided core list is ambiguous and therefore ignored." << std::endl;
        return;
    }

#ifdef __linux__
    // set a cpu mask to prevent moving of threads
    cpu_set_t mask;

    // no CPUs selected
    CPU_ZERO(&mask);

    // no proc_bind
    for(auto it = core_list.begin(); it != core_list.end(); it++)
        CPU_SET(*it, &mask);
    const int set_result = sched_setaffinity(0, sizeof(cpu_set_t), &mask);
#else
    if (!core_list.empty()) {
        std::cout << "WARNING: manipulation of CPU masks is only available for linux systems." << std::endl;
    }
#endif
}
