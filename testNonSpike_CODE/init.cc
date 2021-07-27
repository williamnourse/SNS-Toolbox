#include "definitionsInternal.h"
// ------------------------------------------------------------------------
// merged group arrays
// ------------------------------------------------------------------------
static MergedNeuronInitGroup0 mergedNeuronInitGroup0[1];
static MergedNeuronInitGroup1 mergedNeuronInitGroup1[1];

// ------------------------------------------------------------------------
// merged group functions
// ------------------------------------------------------------------------
void pushMergedNeuronInitGroup0ToDevice(const MergedNeuronInitGroup0 *group) {
    std::copy_n(group, 1, mergedNeuronInitGroup0);
}
void pushMergedNeuronInitGroup1ToDevice(const MergedNeuronInitGroup1 *group) {
    std::copy_n(group, 1, mergedNeuronInitGroup1);
}

// ------------------------------------------------------------------------
// merged extra global parameter functions
// ------------------------------------------------------------------------
void initialize() {
    // ------------------------------------------------------------------------
    // Local neuron groups
     {
        // merged neuron init group 0
        for(unsigned int g = 0; g < 1; g++) {
            const auto &group = mergedNeuronInitGroup0[g]; 
            group.spkCnt[0] = 0;
            for (unsigned i = 0; i < (group.numNeurons); i++) {
                group.spk[i] = 0;
            }
            // current source variables
        }
    }
     {
        // merged neuron init group 1
        for(unsigned int g = 0; g < 1; g++) {
            const auto &group = mergedNeuronInitGroup1[g]; 
            group.spkCnt[0] = 0;
            for (unsigned i = 0; i < (group.numNeurons); i++) {
                group.spk[i] = 0;
            }
            // current source variables
        }
    }
    // ------------------------------------------------------------------------
    // Synapse groups with dense connectivity
    // ------------------------------------------------------------------------
    // Synapse groups with sparse connectivity
}

void initializeSparse() {
    // ------------------------------------------------------------------------
    // Synapse groups with sparse connectivity
}
