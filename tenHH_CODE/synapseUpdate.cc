#include "definitionsInternal.h"
#include "supportCode.h"

// ------------------------------------------------------------------------
// merged group arrays
// ------------------------------------------------------------------------
static MergedPresynapticUpdateGroup0 mergedPresynapticUpdateGroup0[1];
static MergedPresynapticUpdateGroup1 mergedPresynapticUpdateGroup1[1];

// ------------------------------------------------------------------------
// merged group functions
// ------------------------------------------------------------------------
void pushMergedPresynapticUpdateGroup0ToDevice(const MergedPresynapticUpdateGroup0 *group) {
    std::copy_n(group, 1, mergedPresynapticUpdateGroup0);
}
void pushMergedPresynapticUpdateGroup1ToDevice(const MergedPresynapticUpdateGroup1 *group) {
    std::copy_n(group, 1, mergedPresynapticUpdateGroup1);
}

// ------------------------------------------------------------------------
// merged extra global parameter functions
// ------------------------------------------------------------------------
void updateSynapses(float t) {
     {
        // merged presynaptic update group 0
        for(unsigned int g = 0; g < 1; g++) {
            const auto &group = mergedPresynapticUpdateGroup0[g]; 
            const unsigned int postReadDelayOffset = (*group.trgSpkQuePtr) * group.numTrgNeurons;
            // process presynaptic events: True Spikes
            for (unsigned int i = 0; i < group.srcSpkCnt[0]; i++) {
                const unsigned int ipre = group.srcSpk[i];
                const unsigned int npost = group.rowLength[ipre];
                for (unsigned int j = 0; j < npost; j++) {
                    const unsigned int synAddress = (ipre * group.rowStride) + j;
                    const unsigned int ipost = group.ind[synAddress];
                    group.inSyn[ipost] += (-2.00000000000000011e-01f);
                }
            }
            
        }
    }
     {
        // merged presynaptic update group 1
        for(unsigned int g = 0; g < 1; g++) {
            const auto &group = mergedPresynapticUpdateGroup1[g]; 
            const unsigned int preReadDelaySlot = ((*group.srcSpkQuePtr + 1) % 11);
            const unsigned int preReadDelayOffset = preReadDelaySlot * group.numSrcNeurons;
            const unsigned int postReadDelayOffset = (*group.trgSpkQuePtr) * group.numTrgNeurons;
            // process presynaptic events: True Spikes
            for (unsigned int i = 0; i < group.srcSpkCnt[preReadDelaySlot]; i++) {
                const unsigned int ipre = group.srcSpk[preReadDelayOffset + i];
                const unsigned int npost = group.rowLength[ipre];
                for (unsigned int j = 0; j < npost; j++) {
                    const unsigned int synAddress = (ipre * group.rowStride) + j;
                    const unsigned int ipost = group.ind[synAddress];
                    group.inSyn[ipost] += (-2.00000000000000011e-01f);
                }
            }
            
        }
    }
}
