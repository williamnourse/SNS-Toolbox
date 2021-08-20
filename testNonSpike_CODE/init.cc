#include "definitionsInternal.h"
// ------------------------------------------------------------------------
// merged group arrays
// ------------------------------------------------------------------------
static MergedNeuronInitGroup0 mergedNeuronInitGroup0[1];

// ------------------------------------------------------------------------
// merged group functions
// ------------------------------------------------------------------------
void pushMergedNeuronInitGroup0ToDevice(const MergedNeuronInitGroup0 *group) {
    std::copy_n(group, 1, mergedNeuronInitGroup0);
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
             {
                for (unsigned i = 0; i < (group.numNeurons); i++) {
                    group.Val[i] = (0.00000000000000000e+00f);
                }
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
