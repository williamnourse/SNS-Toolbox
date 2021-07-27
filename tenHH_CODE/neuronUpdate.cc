#include "definitionsInternal.h"
#include "supportCode.h"

// ------------------------------------------------------------------------
// merged group arrays
// ------------------------------------------------------------------------
static MergedNeuronSpikeQueueUpdateGroup0 mergedNeuronSpikeQueueUpdateGroup0[1];
static MergedNeuronSpikeQueueUpdateGroup1 mergedNeuronSpikeQueueUpdateGroup1[1];

// ------------------------------------------------------------------------
// merged group functions
// ------------------------------------------------------------------------
void pushMergedNeuronSpikeQueueUpdateGroup0ToDevice(const MergedNeuronSpikeQueueUpdateGroup0 *group) {
    std::copy_n(group, 1, mergedNeuronSpikeQueueUpdateGroup0);
}
void pushMergedNeuronSpikeQueueUpdateGroup1ToDevice(const MergedNeuronSpikeQueueUpdateGroup1 *group) {
    std::copy_n(group, 1, mergedNeuronSpikeQueueUpdateGroup1);
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
void pushMergedNeuronUpdate0spikeTimesToDevice(unsigned int idx, scalar* value) {
    mergedNeuronUpdateGroup0[idx].spikeTimes = value;
}

void updateNeurons(float t) {
     {
        // merged neuron spike queue update group 0
        for(unsigned int g = 0; g < 1; g++) {
            const auto &group = mergedNeuronSpikeQueueUpdateGroup0[g]; 
            group.spkCnt[0] = 0;
        }
    }
     {
        // merged neuron spike queue update group 1
        for(unsigned int g = 0; g < 1; g++) {
            const auto &group = mergedNeuronSpikeQueueUpdateGroup1[g]; 
            group.spkCnt[*group.spkQuePtr] = 0;
        }
    }
     {
        // merged neuron update group 0
        for(unsigned int g = 0; g < 1; g++) {
            const auto &group = mergedNeuronUpdateGroup0[g]; 
            
            for(unsigned int i = 0; i < group.numNeurons; i++) {
                unsigned int lstartSpike = group.startSpike[i];
                const unsigned int lendSpike = group.endSpike[i];
                
                // test whether spike condition was fulfilled previously
                // calculate membrane potential
                
                // test for and register a true spike
                if (lstartSpike != lendSpike && t >= group.spikeTimes[lstartSpike]) {
                    group.spk[group.spkCnt[0]++] = i;
                    // spike reset code
                    lstartSpike++;
                    
                }
                group.startSpike[i] = lstartSpike;
            }
        }
    }
     {
        // merged neuron update group 1
        for(unsigned int g = 0; g < 1; g++) {
            const auto &group = mergedNeuronUpdateGroup1[g]; 
            const unsigned int readDelayOffset = (((*group.spkQuePtr + 10) % 11) * group.numNeurons);
            const unsigned int writeDelayOffset = (*group.spkQuePtr * group.numNeurons);
            
            for(unsigned int i = 0; i < group.numNeurons; i++) {
                scalar lV = group.V[i];
                scalar lm = group.m[i];
                scalar lh = group.h[i];
                scalar ln = group.n[i];
                
                float Isyn = 0;
                 {
                    // pull inSyn values in a coalesced access
                    float linSyn = group.inSynInSyn0[i];
                    Isyn += linSyn * ((-8.00000000000000000e+01f) - lV);
                    linSyn*=(9.04837418035959518e-01f);
                    group.inSynInSyn0[i] = linSyn;
                }
                 {
                    // pull inSyn values in a coalesced access
                    float linSyn = group.inSynInSyn1[i];
                    Isyn += linSyn * ((-8.00000000000000000e+01f) - lV);
                    linSyn*=(9.04837418035959518e-01f);
                    group.inSynInSyn1[i] = linSyn;
                }
                // test whether spike condition was fulfilled previously
                const bool oldSpike= (lV >= 0.0f);
                // calculate membrane potential
                scalar Imem;
                unsigned int mt;
                scalar mdt= DT/25.0f;
                for (mt=0; mt < 25; mt++) {
                   Imem= -(lm*lm*lm*lh*(7.15000000000000036e+00f)*(lV-((5.00000000000000000e+01f)))+
                       ln*ln*ln*ln*(1.42999999999999994e+00f)*(lV-((-9.50000000000000000e+01f)))+
                       (2.67200000000000007e-02f)*(lV-((-6.35630000000000024e+01f)))-Isyn);
                   scalar _a;
                   if (lV == -52.0f) {
                       _a= 1.28f;
                   }
                   else {
                       _a= 0.32f*(-52.0f-lV)/(expf((-52.0f-lV)/4.0f)-1.0f);
                   }
                   scalar _b;
                   if (lV == -25.0f) {
                       _b= 1.4f;
                   }
                   else {
                       _b= 0.28f*(lV+25.0f)/(expf((lV+25.0f)/5.0f)-1.0f);
                   }
                   lm+= (_a*(1.0f-lm)-_b*lm)*mdt;
                   _a= 0.128f*expf((-48.0f-lV)/18.0f);
                   _b= 4.0f / (expf((-25.0f-lV)/5.0f)+1.0f);
                   lh+= (_a*(1.0f-lh)-_b*lh)*mdt;
                   if (lV == -50.0f) {
                       _a= 0.16f;
                   }
                   else {
                       _a= 0.032f*(-50.0f-lV)/(expf((-50.0f-lV)/5.0f)-1.0f);
                   }
                   _b= 0.5f*expf((-55.0f-lV)/40.0f);
                   ln+= (_a*(1.0f-ln)-_b*ln)*mdt;
                   lV+= Imem/(1.42999999999999988e-01f)*mdt;
                }
                
                // test for and register a true spike
                if ((lV >= 0.0f) && !(oldSpike)) {
                    group.spk[writeDelayOffset + group.spkCnt[*group.spkQuePtr]++] = i;
                }
                group.V[i] = lV;
                group.m[i] = lm;
                group.h[i] = lh;
                group.n[i] = ln;
            }
        }
    }
}
