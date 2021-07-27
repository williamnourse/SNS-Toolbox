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
// ------------------------------------------------------------------------
// merged group arrays
// ------------------------------------------------------------------------
static MergedSynapseConnectivityInitGroup0 mergedSynapseConnectivityInitGroup0[1];
static MergedSynapseConnectivityInitGroup1 mergedSynapseConnectivityInitGroup1[1];

// ------------------------------------------------------------------------
// merged group functions
// ------------------------------------------------------------------------
void pushMergedSynapseConnectivityInitGroup0ToDevice(const MergedSynapseConnectivityInitGroup0 *group) {
    std::copy_n(group, 1, mergedSynapseConnectivityInitGroup0);
}
void pushMergedSynapseConnectivityInitGroup1ToDevice(const MergedSynapseConnectivityInitGroup1 *group) {
    std::copy_n(group, 1, mergedSynapseConnectivityInitGroup1);
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
            for (unsigned int d = 0; d < 11; d++) {
                group.spkCnt[d] = 0;
            }
            for (unsigned i = 0; i < (group.numNeurons); i++) {
                for (unsigned int d = 0; d < 11; d++) {
                    group.spk[(d * group.numNeurons) + i] = 0;
                }
            }
             {
                for (unsigned i = 0; i < (group.numNeurons); i++) {
                    group.V[i] = (-6.00000000000000000e+01f);
                }
            }
             {
                for (unsigned i = 0; i < (group.numNeurons); i++) {
                    group.m[i] = (5.29323999999999975e-02f);
                }
            }
             {
                for (unsigned i = 0; i < (group.numNeurons); i++) {
                    group.h[i] = (3.17676699999999979e-01f);
                }
            }
             {
                for (unsigned i = 0; i < (group.numNeurons); i++) {
                    group.n[i] = (5.96120699999999948e-01f);
                }
            }
             {
                for (unsigned i = 0; i < (group.numNeurons); i++) {
                    group.inSynInSyn0[i] = 0.000000f;
                }
            }
             {
                for (unsigned i = 0; i < (group.numNeurons); i++) {
                    group.inSynInSyn1[i] = 0.000000f;
                }
            }
            // current source variables
        }
    }
    // ------------------------------------------------------------------------
    // Synapse groups with dense connectivity
    // ------------------------------------------------------------------------
    // Synapse groups with sparse connectivity
     {
        // merged synapse connectivity init group 0
        for(unsigned int g = 0; g < 1; g++) {
            const auto &group = mergedSynapseConnectivityInitGroup0[g]; 
            memset(group.rowLength, 0, group.numSrcNeurons * sizeof(unsigned int));
            for (unsigned int i = 0; i < group.numSrcNeurons; i++) {
                // Build sparse connectivity
                while(true) {
                    group.ind[(i * group.rowStride) + (group.rowLength[i]++)] = i;
                    break;
                    
                }
            }
        }
    }
     {
        // merged synapse connectivity init group 1
        for(unsigned int g = 0; g < 1; g++) {
            const auto &group = mergedSynapseConnectivityInitGroup1[g]; 
            memset(group.rowLength, 0, group.numSrcNeurons * sizeof(unsigned int));
            for (unsigned int i = 0; i < group.numSrcNeurons; i++) {
                // Build sparse connectivity
                while(true) {
                    
                    group.ind[(i * group.rowStride) + (group.rowLength[i]++)] = (i + 1)%group.numTrgNeurons;
                    break;
                    
                }
            }
        }
    }
}

void initializeSparse() {
    // ------------------------------------------------------------------------
    // Synapse groups with sparse connectivity
}
