#include "definitionsInternal.h"

template<class T>
T *getSymbolAddress(T &devSymbol) {
    return &devSymbol;
}

extern "C" {
// ------------------------------------------------------------------------
// global variables
// ------------------------------------------------------------------------
unsigned long long iT;
float t;

// ------------------------------------------------------------------------
// timers
// ------------------------------------------------------------------------
double neuronUpdateTime;
double initTime;
double presynapticUpdateTime;
double postsynapticUpdateTime;
double synapseDynamicsTime;
double initSparseTime;
// ------------------------------------------------------------------------
// merged group arrays
// ------------------------------------------------------------------------
MergedNeuronInitGroup0 mergedNeuronInitGroup0[1];
MergedNeuronInitGroup1 mergedNeuronInitGroup1[1];
MergedNeuronUpdateGroup0 mergedNeuronUpdateGroup0[1];
MergedNeuronUpdateGroup1 mergedNeuronUpdateGroup1[1];
MergedNeuronSpikeQueueUpdateGroup0 mergedNeuronSpikeQueueUpdateGroup0[2];
// ------------------------------------------------------------------------
// local neuron groups
// ------------------------------------------------------------------------
unsigned int* glbSpkCntInputs;
unsigned int* glbSpkInputs;
scalar* ValInputs;
unsigned int* glbSpkCntPopulation;
unsigned int* glbSpkPopulation;
scalar* UPopulation;
float IappPopulation;

// ------------------------------------------------------------------------
// postsynaptic variables
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// synapse connectivity
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// synapse variables
// ------------------------------------------------------------------------

}  // extern "C"
// ------------------------------------------------------------------------
// extra global params
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// copying things to device
// ------------------------------------------------------------------------
void pushInputsSpikesToDevice(bool uninitialisedOnly) {
}

void pushInputsCurrentSpikesToDevice(bool uninitialisedOnly) {
}

void pushValInputsToDevice(bool uninitialisedOnly) {
}

void pushCurrentValInputsToDevice(bool uninitialisedOnly) {
}

void pushInputsStateToDevice(bool uninitialisedOnly) {
    pushValInputsToDevice(uninitialisedOnly);
}

void pushPopulationSpikesToDevice(bool uninitialisedOnly) {
}

void pushPopulationCurrentSpikesToDevice(bool uninitialisedOnly) {
}

void pushUPopulationToDevice(bool uninitialisedOnly) {
}

void pushCurrentUPopulationToDevice(bool uninitialisedOnly) {
}

void pushPopulationStateToDevice(bool uninitialisedOnly) {
    pushUPopulationToDevice(uninitialisedOnly);
}


// ------------------------------------------------------------------------
// copying things from device
// ------------------------------------------------------------------------
void pullInputsSpikesFromDevice() {
}

void pullInputsCurrentSpikesFromDevice() {
}

void pullValInputsFromDevice() {
}

void pullCurrentValInputsFromDevice() {
}

void pullInputsStateFromDevice() {
    pullValInputsFromDevice();
}

void pullPopulationSpikesFromDevice() {
}

void pullPopulationCurrentSpikesFromDevice() {
}

void pullUPopulationFromDevice() {
}

void pullCurrentUPopulationFromDevice() {
}

void pullPopulationStateFromDevice() {
    pullUPopulationFromDevice();
}


// ------------------------------------------------------------------------
// helper getter functions
// ------------------------------------------------------------------------
unsigned int* getInputsCurrentSpikes() {
    return  glbSpkInputs;
}

unsigned int& getInputsCurrentSpikeCount() {
    return glbSpkCntInputs[0];
}

scalar* getCurrentValInputs() {
    return ValInputs;
}

unsigned int* getPopulationCurrentSpikes() {
    return  glbSpkPopulation;
}

unsigned int& getPopulationCurrentSpikeCount() {
    return glbSpkCntPopulation[0];
}

scalar* getCurrentUPopulation() {
    return UPopulation;
}


void copyStateToDevice(bool uninitialisedOnly) {
    pushInputsStateToDevice(uninitialisedOnly);
    pushPopulationStateToDevice(uninitialisedOnly);
}

void copyConnectivityToDevice(bool uninitialisedOnly) {
}

void copyStateFromDevice() {
    pullInputsStateFromDevice();
    pullPopulationStateFromDevice();
}

void copyCurrentSpikesFromDevice() {
    pullInputsCurrentSpikesFromDevice();
    pullPopulationCurrentSpikesFromDevice();
}

void copyCurrentSpikeEventsFromDevice() {
}

void allocateMem() {
    // ------------------------------------------------------------------------
    // global variables
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // timers
    // ------------------------------------------------------------------------
    // ------------------------------------------------------------------------
    // local neuron groups
    // ------------------------------------------------------------------------
    glbSpkCntInputs = new unsigned int[1];
    glbSpkInputs = new unsigned int[2];
    ValInputs = new scalar[2];
    glbSpkCntPopulation = new unsigned int[1];
    glbSpkPopulation = new unsigned int[6];
    UPopulation = new scalar[6];
    
    // ------------------------------------------------------------------------
    // postsynaptic variables
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // synapse connectivity
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // synapse variables
    // ------------------------------------------------------------------------
    
    mergedNeuronInitGroup0[0] = {glbSpkCntPopulation, glbSpkPopulation, 6, };
    pushMergedNeuronInitGroup0ToDevice(mergedNeuronInitGroup0);
    mergedNeuronInitGroup1[0] = {glbSpkCntInputs, glbSpkInputs, 2, };
    pushMergedNeuronInitGroup1ToDevice(mergedNeuronInitGroup1);
    mergedNeuronUpdateGroup0[0] = {glbSpkCntPopulation, glbSpkPopulation, UPopulation, 6, IappPopulation, };
    pushMergedNeuronUpdateGroup0ToDevice(mergedNeuronUpdateGroup0);
    mergedNeuronUpdateGroup1[0] = {glbSpkCntInputs, glbSpkInputs, ValInputs, 2, };
    pushMergedNeuronUpdateGroup1ToDevice(mergedNeuronUpdateGroup1);
    mergedNeuronSpikeQueueUpdateGroup0[0] = {glbSpkCntPopulation, };
    mergedNeuronSpikeQueueUpdateGroup0[1] = {glbSpkCntInputs, };
    pushMergedNeuronSpikeQueueUpdateGroup0ToDevice(mergedNeuronSpikeQueueUpdateGroup0);
}

void freeMem() {
    // ------------------------------------------------------------------------
    // global variables
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // timers
    // ------------------------------------------------------------------------
    // ------------------------------------------------------------------------
    // local neuron groups
    // ------------------------------------------------------------------------
    delete[] glbSpkCntInputs;
    delete[] glbSpkInputs;
    delete[] ValInputs;
    delete[] glbSpkCntPopulation;
    delete[] glbSpkPopulation;
    delete[] UPopulation;
    
    // ------------------------------------------------------------------------
    // postsynaptic variables
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // synapse connectivity
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // synapse variables
    // ------------------------------------------------------------------------
    
}

size_t getFreeDeviceMemBytes() {
    return 0;
}

void stepTime() {
    updateSynapses(t);
    updateNeurons(t);
    iT++;
    t = iT*DT;
}

