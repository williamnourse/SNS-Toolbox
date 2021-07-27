#include "definitionsInternal.h"
#include "supportCode.h"

// ------------------------------------------------------------------------
// merged group arrays
// ------------------------------------------------------------------------
static MergedNeuronSpikeQueueUpdateGroup0 mergedNeuronSpikeQueueUpdateGroup0[2];

// ------------------------------------------------------------------------
// merged group functions
// ------------------------------------------------------------------------
void pushMergedNeuronSpikeQueueUpdateGroup0ToDevice(const MergedNeuronSpikeQueueUpdateGroup0 *group) {
    std::copy_n(group, 2, mergedNeuronSpikeQueueUpdateGroup0);
}

// ------------------------------------------------------------------------
// merged extra global parameter functions
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// merged group arrays
// ------------------------------------------------------------------------
static MergedNeuronUpdateGroup0 mergedNeuronUpdateGroup0[1];
static MergedNeuronUpdateGroup1 mergedNeuronUpdateGroup1[1];

// ------------------------------------------------------------------------
// merged group functions
// ------------------------------------------------------------------------
void pushMergedNeuronUpdateGroup0ToDevice(const MergedNeuronUpdateGroup0 *group) {
    std::copy_n(group, 1, mergedNeuronUpdateGroup0);
}
void pushMergedNeuronUpdateGroup1ToDevice(const MergedNeuronUpdateGroup1 *group) {
    std::copy_n(group, 1, mergedNeuronUpdateGroup1);
}

// ------------------------------------------------------------------------
// merged extra global parameter functions
// ------------------------------------------------------------------------
void updateNeurons(float t) {
    mergedNeuronUpdateGroup0[0].Iapp = IappPopulation;
     {
        // merged neuron spike queue update group 0
        for(unsigned int g = 0; g < 2; g++) {
            const auto &group = mergedNeuronSpikeQueueUpdateGroup0[g]; 
            group.spkCnt[0] = 0;
        }
    }
     {
        // merged neuron update group 0
        for(unsigned int g = 0; g < 1; g++) {
            const auto &group = mergedNeuronUpdateGroup0[g]; 
            
            for(unsigned int i = 0; i < group.numNeurons; i++) {
                scalar lU = group.U[i];
                
                float Isyn = 0;
                // calculate membrane potential
                lU+= (-(1.00000000000000000e+00f)*lU + (0.00000000000000000e+00f) + group.Iapp + Isyn)*(DT/(5.00000000000000000e+00f));
                group.U[i] = lU;
            }
        }
    }
     {
        // merged neuron update group 1
        for(unsigned int g = 0; g < 1; g++) {
            const auto &group = mergedNeuronUpdateGroup1[g]; 
            
            for(unsigned int i = 0; i < group.numNeurons; i++) {
                scalar lVal = group.Val[i];
                
                // calculate membrane potential
                
                group.Val[i] = lVal;
            }
        }
    }
}
