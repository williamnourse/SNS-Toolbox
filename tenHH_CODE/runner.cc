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
MergedSynapseConnectivityInitGroup0 mergedSynapseConnectivityInitGroup0[1];
MergedSynapseConnectivityInitGroup1 mergedSynapseConnectivityInitGroup1[1];
MergedNeuronUpdateGroup0 mergedNeuronUpdateGroup0[1];
MergedNeuronUpdateGroup1 mergedNeuronUpdateGroup1[1];
MergedPresynapticUpdateGroup0 mergedPresynapticUpdateGroup0[1];
MergedPresynapticUpdateGroup1 mergedPresynapticUpdateGroup1[1];
MergedNeuronSpikeQueueUpdateGroup0 mergedNeuronSpikeQueueUpdateGroup0[1];
MergedNeuronSpikeQueueUpdateGroup1 mergedNeuronSpikeQueueUpdateGroup1[1];
// ------------------------------------------------------------------------
// local neuron groups
// ------------------------------------------------------------------------
unsigned int* glbSpkCntPop1;
unsigned int* glbSpkPop1;
unsigned int spkQuePtrPop1;
scalar* VPop1;
scalar* mPop1;
scalar* hPop1;
scalar* nPop1;
unsigned int* glbSpkCntStim;
unsigned int* glbSpkStim;
unsigned int* startSpikeStim;
unsigned int* endSpikeStim;
scalar* spikeTimesStim;

// ------------------------------------------------------------------------
// postsynaptic variables
// ------------------------------------------------------------------------
float* inSynStimPop1;
float* inSynPop1self;

// ------------------------------------------------------------------------
// synapse connectivity
// ------------------------------------------------------------------------
const unsigned int maxRowLengthPop1self = 1;
unsigned int* rowLengthPop1self;
uint32_t* indPop1self;
const unsigned int maxRowLengthStimPop1 = 1;
unsigned int* rowLengthStimPop1;
uint32_t* indStimPop1;

// ------------------------------------------------------------------------
// synapse variables
// ------------------------------------------------------------------------

}  // extern "C"
// ------------------------------------------------------------------------
// extra global params
// ------------------------------------------------------------------------
void allocatespikeTimesStim(unsigned int count) {
    spikeTimesStim = new scalar[count];
    pushMergedNeuronUpdate0spikeTimesToDevice(0, spikeTimesStim);
}
void freespikeTimesStim() {
    delete[] spikeTimesStim;
}
void pushspikeTimesStimToDevice(unsigned int count) {
}
void pullspikeTimesStimFromDevice(unsigned int count) {
}

// ------------------------------------------------------------------------
// copying things to device
// ------------------------------------------------------------------------
void pushPop1SpikesToDevice(bool uninitialisedOnly) {
}

void pushPop1CurrentSpikesToDevice(bool uninitialisedOnly) {
}

void pushVPop1ToDevice(bool uninitialisedOnly) {
}

void pushCurrentVPop1ToDevice(bool uninitialisedOnly) {
}

void pushmPop1ToDevice(bool uninitialisedOnly) {
}

void pushCurrentmPop1ToDevice(bool uninitialisedOnly) {
}

void pushhPop1ToDevice(bool uninitialisedOnly) {
}

void pushCurrenthPop1ToDevice(bool uninitialisedOnly) {
}

void pushnPop1ToDevice(bool uninitialisedOnly) {
}

void pushCurrentnPop1ToDevice(bool uninitialisedOnly) {
}

void pushPop1StateToDevice(bool uninitialisedOnly) {
    pushVPop1ToDevice(uninitialisedOnly);
    pushmPop1ToDevice(uninitialisedOnly);
    pushhPop1ToDevice(uninitialisedOnly);
    pushnPop1ToDevice(uninitialisedOnly);
}

void pushStimSpikesToDevice(bool uninitialisedOnly) {
}

void pushStimCurrentSpikesToDevice(bool uninitialisedOnly) {
}

void pushstartSpikeStimToDevice(bool uninitialisedOnly) {
}

void pushCurrentstartSpikeStimToDevice(bool uninitialisedOnly) {
}

void pushendSpikeStimToDevice(bool uninitialisedOnly) {
}

void pushCurrentendSpikeStimToDevice(bool uninitialisedOnly) {
}

void pushStimStateToDevice(bool uninitialisedOnly) {
    pushstartSpikeStimToDevice(uninitialisedOnly);
    pushendSpikeStimToDevice(uninitialisedOnly);
}

void pushPop1selfConnectivityToDevice(bool uninitialisedOnly) {
}

void pushStimPop1ConnectivityToDevice(bool uninitialisedOnly) {
}

void pushinSynPop1selfToDevice(bool uninitialisedOnly) {
}

void pushPop1selfStateToDevice(bool uninitialisedOnly) {
    pushinSynPop1selfToDevice(uninitialisedOnly);
}

void pushinSynStimPop1ToDevice(bool uninitialisedOnly) {
}

void pushStimPop1StateToDevice(bool uninitialisedOnly) {
    pushinSynStimPop1ToDevice(uninitialisedOnly);
}


// ------------------------------------------------------------------------
// copying things from device
// ------------------------------------------------------------------------
void pullPop1SpikesFromDevice() {
}

void pullPop1CurrentSpikesFromDevice() {
}

void pullVPop1FromDevice() {
}

void pullCurrentVPop1FromDevice() {
}

void pullmPop1FromDevice() {
}

void pullCurrentmPop1FromDevice() {
}

void pullhPop1FromDevice() {
}

void pullCurrenthPop1FromDevice() {
}

void pullnPop1FromDevice() {
}

void pullCurrentnPop1FromDevice() {
}

void pullPop1StateFromDevice() {
    pullVPop1FromDevice();
    pullmPop1FromDevice();
    pullhPop1FromDevice();
    pullnPop1FromDevice();
}

void pullStimSpikesFromDevice() {
}

void pullStimCurrentSpikesFromDevice() {
}

void pullstartSpikeStimFromDevice() {
}

void pullCurrentstartSpikeStimFromDevice() {
}

void pullendSpikeStimFromDevice() {
}

void pullCurrentendSpikeStimFromDevice() {
}

void pullStimStateFromDevice() {
    pullstartSpikeStimFromDevice();
    pullendSpikeStimFromDevice();
}

void pullPop1selfConnectivityFromDevice() {
}

void pullStimPop1ConnectivityFromDevice() {
}

void pullinSynPop1selfFromDevice() {
}

void pullPop1selfStateFromDevice() {
    pullinSynPop1selfFromDevice();
}

void pullinSynStimPop1FromDevice() {
}

void pullStimPop1StateFromDevice() {
    pullinSynStimPop1FromDevice();
}


// ------------------------------------------------------------------------
// helper getter functions
// ------------------------------------------------------------------------
unsigned int* getPop1CurrentSpikes() {
    return  (glbSpkPop1 + (spkQuePtrPop1 * 10));
}

unsigned int& getPop1CurrentSpikeCount() {
    return glbSpkCntPop1[spkQuePtrPop1];
}

scalar* getCurrentVPop1() {
    return VPop1;
}

scalar* getCurrentmPop1() {
    return mPop1;
}

scalar* getCurrenthPop1() {
    return hPop1;
}

scalar* getCurrentnPop1() {
    return nPop1;
}

unsigned int* getStimCurrentSpikes() {
    return  glbSpkStim;
}

unsigned int& getStimCurrentSpikeCount() {
    return glbSpkCntStim[0];
}

unsigned int* getCurrentstartSpikeStim() {
    return startSpikeStim;
}

unsigned int* getCurrentendSpikeStim() {
    return endSpikeStim;
}


void copyStateToDevice(bool uninitialisedOnly) {
    pushPop1StateToDevice(uninitialisedOnly);
    pushStimStateToDevice(uninitialisedOnly);
    pushPop1selfStateToDevice(uninitialisedOnly);
    pushStimPop1StateToDevice(uninitialisedOnly);
}

void copyConnectivityToDevice(bool uninitialisedOnly) {
    pushPop1selfConnectivityToDevice(uninitialisedOnly);
    pushStimPop1ConnectivityToDevice(uninitialisedOnly);
}

void copyStateFromDevice() {
    pullPop1StateFromDevice();
    pullStimStateFromDevice();
    pullPop1selfStateFromDevice();
    pullStimPop1StateFromDevice();
}

void copyCurrentSpikesFromDevice() {
    pullPop1CurrentSpikesFromDevice();
    pullStimCurrentSpikesFromDevice();
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
    glbSpkCntPop1 = new unsigned int[11];
    glbSpkPop1 = new unsigned int[110];
    VPop1 = new scalar[10];
    mPop1 = new scalar[10];
    hPop1 = new scalar[10];
    nPop1 = new scalar[10];
    glbSpkCntStim = new unsigned int[1];
    glbSpkStim = new unsigned int[1];
    startSpikeStim = new unsigned int[1];
    endSpikeStim = new unsigned int[1];
    
    // ------------------------------------------------------------------------
    // postsynaptic variables
    // ------------------------------------------------------------------------
    inSynStimPop1 = new float[10];
    inSynPop1self = new float[10];
    
    // ------------------------------------------------------------------------
    // synapse connectivity
    // ------------------------------------------------------------------------
    rowLengthPop1self = new unsigned int[10];
    indPop1self = new uint32_t[10];
    rowLengthStimPop1 = new unsigned int[1];
    indStimPop1 = new uint32_t[1];
    
    // ------------------------------------------------------------------------
    // synapse variables
    // ------------------------------------------------------------------------
    
    mergedNeuronInitGroup0[0] = {glbSpkCntStim, glbSpkStim, 1, };
    pushMergedNeuronInitGroup0ToDevice(mergedNeuronInitGroup0);
    mergedNeuronInitGroup1[0] = {glbSpkCntPop1, glbSpkPop1, getSymbolAddress(spkQuePtrPop1), VPop1, mPop1, hPop1, nPop1, inSynStimPop1, inSynPop1self, 10, };
    pushMergedNeuronInitGroup1ToDevice(mergedNeuronInitGroup1);
    mergedSynapseConnectivityInitGroup0[0] = {rowLengthStimPop1, indStimPop1, 1, 10, 1, };
    pushMergedSynapseConnectivityInitGroup0ToDevice(mergedSynapseConnectivityInitGroup0);
    mergedSynapseConnectivityInitGroup1[0] = {rowLengthPop1self, indPop1self, 10, 10, 1, };
    pushMergedSynapseConnectivityInitGroup1ToDevice(mergedSynapseConnectivityInitGroup1);
    mergedNeuronUpdateGroup0[0] = {glbSpkCntStim, glbSpkStim, startSpikeStim, endSpikeStim, spikeTimesStim, 1, };
    pushMergedNeuronUpdateGroup0ToDevice(mergedNeuronUpdateGroup0);
    mergedNeuronUpdateGroup1[0] = {glbSpkCntPop1, glbSpkPop1, getSymbolAddress(spkQuePtrPop1), VPop1, mPop1, hPop1, nPop1, inSynStimPop1, inSynPop1self, 10, };
    pushMergedNeuronUpdateGroup1ToDevice(mergedNeuronUpdateGroup1);
    mergedPresynapticUpdateGroup0[0] = {inSynStimPop1, glbSpkCntStim, glbSpkStim, getSymbolAddress(spkQuePtrPop1), rowLengthStimPop1, indStimPop1, 1, 1, 10, };
    pushMergedPresynapticUpdateGroup0ToDevice(mergedPresynapticUpdateGroup0);
    mergedPresynapticUpdateGroup1[0] = {inSynPop1self, glbSpkCntPop1, glbSpkPop1, getSymbolAddress(spkQuePtrPop1), getSymbolAddress(spkQuePtrPop1), rowLengthPop1self, indPop1self, 1, 10, 10, };
    pushMergedPresynapticUpdateGroup1ToDevice(mergedPresynapticUpdateGroup1);
    mergedNeuronSpikeQueueUpdateGroup0[0] = {glbSpkCntStim, };
    pushMergedNeuronSpikeQueueUpdateGroup0ToDevice(mergedNeuronSpikeQueueUpdateGroup0);
    mergedNeuronSpikeQueueUpdateGroup1[0] = {getSymbolAddress(spkQuePtrPop1), glbSpkCntPop1, 11, };
    pushMergedNeuronSpikeQueueUpdateGroup1ToDevice(mergedNeuronSpikeQueueUpdateGroup1);
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
    delete[] glbSpkCntPop1;
    delete[] glbSpkPop1;
    delete[] VPop1;
    delete[] mPop1;
    delete[] hPop1;
    delete[] nPop1;
    delete[] glbSpkCntStim;
    delete[] glbSpkStim;
    delete[] startSpikeStim;
    delete[] endSpikeStim;
    
    // ------------------------------------------------------------------------
    // postsynaptic variables
    // ------------------------------------------------------------------------
    delete[] inSynStimPop1;
    delete[] inSynPop1self;
    
    // ------------------------------------------------------------------------
    // synapse connectivity
    // ------------------------------------------------------------------------
    delete[] rowLengthPop1self;
    delete[] indPop1self;
    delete[] rowLengthStimPop1;
    delete[] indStimPop1;
    
    // ------------------------------------------------------------------------
    // synapse variables
    // ------------------------------------------------------------------------
    
}

size_t getFreeDeviceMemBytes() {
    return 0;
}

void stepTime() {
    updateSynapses(t);
    spkQuePtrPop1 = (spkQuePtrPop1 + 1) % 11;
    updateNeurons(t);
    iT++;
    t = iT*DT;
}

