#pragma once
#include "definitions.h"

#define SUPPORT_CODE_FUNC inline
#define gennCLZ __builtin_clz

// ------------------------------------------------------------------------
// merged group structures
// ------------------------------------------------------------------------
struct MergedNeuronInitGroup0
 {
    unsigned int* spkCnt;
    unsigned int* spk;
    unsigned int numNeurons;
    
}
;
struct MergedNeuronInitGroup1
 {
    unsigned int* spkCnt;
    unsigned int* spk;
    unsigned int numNeurons;
    
}
;
struct MergedNeuronUpdateGroup0
 {
    unsigned int* spkCnt;
    unsigned int* spk;
    scalar* U;
    unsigned int numNeurons;
    float Iapp;
    
}
;
struct MergedNeuronUpdateGroup1
 {
    unsigned int* spkCnt;
    unsigned int* spk;
    scalar* Val;
    unsigned int numNeurons;
    
}
;
struct MergedNeuronSpikeQueueUpdateGroup0
 {
    unsigned int* spkCnt;
    
}
;
extern "C" {
// ------------------------------------------------------------------------
// global variables
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// timers
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// merged group arrays for host initialisation
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// local neuron groups
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// postsynaptic variables
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// synapse connectivity
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// synapse variables
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// copying merged group structures to device
// ------------------------------------------------------------------------
EXPORT_FUNC void pushMergedNeuronInitGroup0ToDevice(const MergedNeuronInitGroup0 *group);
EXPORT_FUNC void pushMergedNeuronInitGroup1ToDevice(const MergedNeuronInitGroup1 *group);
EXPORT_FUNC void pushMergedNeuronUpdateGroup0ToDevice(const MergedNeuronUpdateGroup0 *group);
EXPORT_FUNC void pushMergedNeuronUpdateGroup1ToDevice(const MergedNeuronUpdateGroup1 *group);
EXPORT_FUNC void pushMergedNeuronSpikeQueueUpdateGroup0ToDevice(const MergedNeuronSpikeQueueUpdateGroup0 *group);
}  // extern "C"
