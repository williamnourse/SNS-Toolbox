#pragma once
#define EXPORT_VAR extern
#define EXPORT_FUNC
// Standard C++ includes
#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>

// Standard C includes
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#define DT 1.00000000000000006e-01f
typedef float scalar;
#define SCALAR_MIN 1.175494351e-38f
#define SCALAR_MAX 3.402823466e+38f

#define TIME_MIN 1.175494351e-38f
#define TIME_MAX 3.402823466e+38f

// ------------------------------------------------------------------------
// bit tool macros
#define B(x,i) ((x) & (0x80000000 >> (i))) //!< Extract the bit at the specified position i from x
#define setB(x,i) x= ((x) | (0x80000000 >> (i))) //!< Set the bit at the specified position i in x to 1
#define delB(x,i) x= ((x) & (~(0x80000000 >> (i)))) //!< Set the bit at the specified position i in x to 0

extern "C" {
// ------------------------------------------------------------------------
// global variables
// ------------------------------------------------------------------------
EXPORT_VAR unsigned long long iT;
EXPORT_VAR float t;

// ------------------------------------------------------------------------
// timers
// ------------------------------------------------------------------------
EXPORT_VAR double neuronUpdateTime;
EXPORT_VAR double initTime;
EXPORT_VAR double presynapticUpdateTime;
EXPORT_VAR double postsynapticUpdateTime;
EXPORT_VAR double synapseDynamicsTime;
EXPORT_VAR double initSparseTime;
// ------------------------------------------------------------------------
// local neuron groups
// ------------------------------------------------------------------------
#define spikeCount_Pop1 glbSpkCntPop1[spkQuePtrPop1]
#define spike_Pop1 (glbSpkPop1 + (spkQuePtrPop1 * 10))
#define glbSpkShiftPop1 spkQuePtrPop1*10

EXPORT_VAR unsigned int* glbSpkCntPop1;
EXPORT_VAR unsigned int* glbSpkPop1;
EXPORT_VAR unsigned int spkQuePtrPop1;
EXPORT_VAR scalar* VPop1;
EXPORT_VAR scalar* mPop1;
EXPORT_VAR scalar* hPop1;
EXPORT_VAR scalar* nPop1;
#define spikeCount_Stim glbSpkCntStim[0]
#define spike_Stim glbSpkStim
#define glbSpkShiftStim 0

EXPORT_VAR unsigned int* glbSpkCntStim;
EXPORT_VAR unsigned int* glbSpkStim;
EXPORT_VAR unsigned int* startSpikeStim;
EXPORT_VAR unsigned int* endSpikeStim;
EXPORT_VAR scalar* spikeTimesStim;

// ------------------------------------------------------------------------
// postsynaptic variables
// ------------------------------------------------------------------------
EXPORT_VAR float* inSynStimPop1;
EXPORT_VAR float* inSynPop1self;

// ------------------------------------------------------------------------
// synapse connectivity
// ------------------------------------------------------------------------
EXPORT_VAR const unsigned int maxRowLengthPop1self;
EXPORT_VAR unsigned int* rowLengthPop1self;
EXPORT_VAR uint32_t* indPop1self;
EXPORT_VAR const unsigned int maxRowLengthStimPop1;
EXPORT_VAR unsigned int* rowLengthStimPop1;
EXPORT_VAR uint32_t* indStimPop1;

// ------------------------------------------------------------------------
// synapse variables
// ------------------------------------------------------------------------

EXPORT_FUNC void pushPop1SpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullPop1SpikesFromDevice();
EXPORT_FUNC void pushPop1CurrentSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullPop1CurrentSpikesFromDevice();
EXPORT_FUNC unsigned int* getPop1CurrentSpikes();
EXPORT_FUNC unsigned int& getPop1CurrentSpikeCount();
EXPORT_FUNC void pushVPop1ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullVPop1FromDevice();
EXPORT_FUNC void pushCurrentVPop1ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentVPop1FromDevice();
EXPORT_FUNC scalar* getCurrentVPop1();
EXPORT_FUNC void pushmPop1ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullmPop1FromDevice();
EXPORT_FUNC void pushCurrentmPop1ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentmPop1FromDevice();
EXPORT_FUNC scalar* getCurrentmPop1();
EXPORT_FUNC void pushhPop1ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullhPop1FromDevice();
EXPORT_FUNC void pushCurrenthPop1ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrenthPop1FromDevice();
EXPORT_FUNC scalar* getCurrenthPop1();
EXPORT_FUNC void pushnPop1ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullnPop1FromDevice();
EXPORT_FUNC void pushCurrentnPop1ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentnPop1FromDevice();
EXPORT_FUNC scalar* getCurrentnPop1();
EXPORT_FUNC void pushPop1StateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullPop1StateFromDevice();
EXPORT_FUNC void pushStimSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullStimSpikesFromDevice();
EXPORT_FUNC void pushStimCurrentSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullStimCurrentSpikesFromDevice();
EXPORT_FUNC unsigned int* getStimCurrentSpikes();
EXPORT_FUNC unsigned int& getStimCurrentSpikeCount();
EXPORT_FUNC void pushstartSpikeStimToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullstartSpikeStimFromDevice();
EXPORT_FUNC void pushCurrentstartSpikeStimToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentstartSpikeStimFromDevice();
EXPORT_FUNC unsigned int* getCurrentstartSpikeStim();
EXPORT_FUNC void pushendSpikeStimToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullendSpikeStimFromDevice();
EXPORT_FUNC void pushCurrentendSpikeStimToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentendSpikeStimFromDevice();
EXPORT_FUNC unsigned int* getCurrentendSpikeStim();
EXPORT_FUNC void pushStimStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullStimStateFromDevice();
EXPORT_FUNC void allocatespikeTimesStim(unsigned int count);
EXPORT_FUNC void freespikeTimesStim();
EXPORT_FUNC void pushspikeTimesStimToDevice(unsigned int count);
EXPORT_FUNC void pullspikeTimesStimFromDevice(unsigned int count);
EXPORT_FUNC void pushPop1selfConnectivityToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullPop1selfConnectivityFromDevice();
EXPORT_FUNC void pushStimPop1ConnectivityToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullStimPop1ConnectivityFromDevice();
EXPORT_FUNC void pushinSynPop1selfToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSynPop1selfFromDevice();
EXPORT_FUNC void pushPop1selfStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullPop1selfStateFromDevice();
EXPORT_FUNC void pushinSynStimPop1ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSynStimPop1FromDevice();
EXPORT_FUNC void pushStimPop1StateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullStimPop1StateFromDevice();
// Runner functions
EXPORT_FUNC void copyStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void copyConnectivityToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void copyStateFromDevice();
EXPORT_FUNC void copyCurrentSpikesFromDevice();
EXPORT_FUNC void copyCurrentSpikeEventsFromDevice();
EXPORT_FUNC void allocateMem();
EXPORT_FUNC void freeMem();
EXPORT_FUNC size_t getFreeDeviceMemBytes();
EXPORT_FUNC void stepTime();

// Functions generated by backend
EXPORT_FUNC void updateNeurons(float t);
EXPORT_FUNC void updateSynapses(float t);
EXPORT_FUNC void initialize();
EXPORT_FUNC void initializeSparse();
}  // extern "C"
