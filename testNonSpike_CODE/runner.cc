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
MergedNeuronUpdateGroup0 mergedNeuronUpdateGroup0[1];
MergedNeuronSpikeQueueUpdateGroup0 mergedNeuronSpikeQueueUpdateGroup0[1];
// ------------------------------------------------------------------------
// local neuron groups
// ------------------------------------------------------------------------
unsigned int* glbSpkCntApplied Current;
unsigned int* glbSpkApplied Current;
scalar* ValApplied Current;

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
void pushApplied CurrentSpikesToDevice(bool uninitialisedOnly) {
}

void pushApplied CurrentCurrentSpikesToDevice(bool uninitialisedOnly) {
}

void pushValApplied CurrentToDevice(bool uninitialisedOnly) {
}

void pushCurrentValApplied CurrentToDevice(bool uninitialisedOnly) {
}

void pushApplied CurrentStateToDevice(bool uninitialisedOnly) {
    pushValApplied CurrentToDevice(uninitialisedOnly);
}


// ------------------------------------------------------------------------
// copying things from device
// ------------------------------------------------------------------------
void pullApplied CurrentSpikesFromDevice() {
}

void pullApplied CurrentCurrentSpikesFromDevice() {
}

void pullValApplied CurrentFromDevice() {
}

void pullCurrentValApplied CurrentFromDevice() {
}

void pullApplied CurrentStateFromDevice() {
    pullValApplied CurrentFromDevice();
}


// ------------------------------------------------------------------------
// helper getter functions
// ------------------------------------------------------------------------
unsigned int* getApplied CurrentCurrentSpikes() {
    return  glbSpkApplied Current;
}

unsigned int& getApplied CurrentCurrentSpikeCount() {
    return glbSpkCntApplied Current[0];
}

scalar* getCurrentValApplied Current() {
    return ValApplied Current;
}


void copyStateToDevice(bool uninitialisedOnly) {
    pushApplied CurrentStateToDevice(uninitialisedOnly);
}

void copyConnectivityToDevice(bool uninitialisedOnly) {
}

void copyStateFromDevice() {
    pullApplied CurrentStateFromDevice();
}

void copyCurrentSpikesFromDevice() {
    pullApplied CurrentCurrentSpikesFromDevice();
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
    glbSpkCntApplied Current = new unsigned int[1];
    glbSpkApplied Current = new unsigned int[1];
    ValApplied Current = new scalar[1];
    
    // ------------------------------------------------------------------------
    // postsynaptic variables
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // synapse connectivity
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // synapse variables
    // ------------------------------------------------------------------------
    
    mergedNeuronInitGroup0[0] = {glbSpkCntApplied Current, glbSpkApplied Current, ValApplied Current, 1, };
    pushMergedNeuronInitGroup0ToDevice(mergedNeuronInitGroup0);
    mergedNeuronUpdateGroup0[0] = {glbSpkCntApplied Current, glbSpkApplied Current, ValApplied Current, 1, };
    pushMergedNeuronUpdateGroup0ToDevice(mergedNeuronUpdateGroup0);
    mergedNeuronSpikeQueueUpdateGroup0[0] = {glbSpkCntApplied Current, };
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
    delete[] glbSpkCntApplied Current;
    delete[] glbSpkApplied Current;
    delete[] ValApplied Current;
    
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

