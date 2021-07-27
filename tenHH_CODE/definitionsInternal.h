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
    volatile unsigned int* spkQuePtr;
    scalar* V;
    scalar* m;
    scalar* h;
    scalar* n;
    float* inSynInSyn0;
    float* inSynInSyn1;
    unsigned int numNeurons;
    
}
;
struct MergedSynapseConnectivityInitGroup0
 {
    unsigned int* rowLength;
    uint32_t* ind;
    unsigned int numSrcNeurons;
    unsigned int numTrgNeurons;
    unsigned int rowStride;
    
}
;
struct MergedSynapseConnectivityInitGroup1
 {
    unsigned int* rowLength;
    uint32_t* ind;
    unsigned int numSrcNeurons;
    unsigned int numTrgNeurons;
    unsigned int rowStride;
    
}
;
struct MergedNeuronUpdateGroup0
 {
    unsigned int* spkCnt;
    unsigned int* spk;
    unsigned int* startSpike;
    unsigned int* endSpike;
    scalar* spikeTimes;
    unsigned int numNeurons;
    
}
;
struct MergedNeuronUpdateGroup1
 {
    unsigned int* spkCnt;
    unsigned int* spk;
    volatile unsigned int* spkQuePtr;
    scalar* V;
    scalar* m;
    scalar* h;
    scalar* n;
    float* inSynInSyn0;
    float* inSynInSyn1;
    unsigned int numNeurons;
    
}
;
struct MergedPresynapticUpdateGroup0
 {
    float* inSyn;
    unsigned int* srcSpkCnt;
    unsigned int* srcSpk;
    volatile unsigned int* trgSpkQuePtr;
    unsigned int* rowLength;
    uint32_t* ind;
    unsigned int rowStride;
    unsigned int numSrcNeurons;
    unsigned int numTrgNeurons;
    
}
;
struct MergedPresynapticUpdateGroup1
 {
    float* inSyn;
    unsigned int* srcSpkCnt;
    unsigned int* srcSpk;
    volatile unsigned int* srcSpkQuePtr;
    volatile unsigned int* trgSpkQuePtr;
    unsigned int* rowLength;
    uint32_t* ind;
    unsigned int rowStride;
    unsigned int numSrcNeurons;
    unsigned int numTrgNeurons;
    
}
;
struct MergedNeuronSpikeQueueUpdateGroup0
 {
    unsigned int* spkCnt;
    
}
;
struct MergedNeuronSpikeQueueUpdateGroup1
 {
    volatile unsigned int* spkQuePtr;
    unsigned int* spkCnt;
    unsigned int numDelaySlots;
    
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
EXPORT_FUNC void pushMergedSynapseConnectivityInitGroup0ToDevice(const MergedSynapseConnectivityInitGroup0 *group);
EXPORT_FUNC void pushMergedSynapseConnectivityInitGroup1ToDevice(const MergedSynapseConnectivityInitGroup1 *group);
EXPORT_FUNC void pushMergedNeuronUpdate0spikeTimesToDevice(unsigned int idx, scalar* value);
EXPORT_FUNC void pushMergedNeuronUpdateGroup0ToDevice(const MergedNeuronUpdateGroup0 *group);
EXPORT_FUNC void pushMergedNeuronUpdateGroup1ToDevice(const MergedNeuronUpdateGroup1 *group);
EXPORT_FUNC void pushMergedPresynapticUpdateGroup0ToDevice(const MergedPresynapticUpdateGroup0 *group);
EXPORT_FUNC void pushMergedPresynapticUpdateGroup1ToDevice(const MergedPresynapticUpdateGroup1 *group);
EXPORT_FUNC void pushMergedNeuronSpikeQueueUpdateGroup0ToDevice(const MergedNeuronSpikeQueueUpdateGroup0 *group);
EXPORT_FUNC void pushMergedNeuronSpikeQueueUpdateGroup1ToDevice(const MergedNeuronSpikeQueueUpdateGroup1 *group);
}  // extern "C"
