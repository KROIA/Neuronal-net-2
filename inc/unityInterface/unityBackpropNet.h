#pragma once
#include "unityConfig.h"
#include "backend/backpropNet.h"

#ifdef MAKE_DLL
#define NET_PARAM NeuronalNet::BackpropNet* net

UNITY_NET_API void* BackpropNet_instantiate();
UNITY_NET_API void BackpropNet_dealocate(NET_PARAM);

UNITY_NET_API bool BackpropNet_build(NET_PARAM);

UNITY_NET_API void BackpropNet_setLearnParameter(NET_PARAM, float learnParam);
UNITY_NET_API float BackpropNet_getLearnParameter(NET_PARAM);

UNITY_NET_API void BackpropNet_setExpectedOutput1(NET_PARAM, const NeuronalNet::MultiSignalVector* expectedOutputVec);
UNITY_NET_API void BackpropNet_setExpectedOutput2(NET_PARAM, const NeuronalNet::SignalVector* expectedOutputVec);
UNITY_NET_API void BackpropNet_setExpectedOutput3(NET_PARAM, size_t streamIndex, const NeuronalNet::SignalVector* expectedOutputVec);
UNITY_NET_API void BackpropNet_learn1(NET_PARAM);
UNITY_NET_API void BackpropNet_learn2(NET_PARAM, size_t streamIndex);
UNITY_NET_API void BackpropNet_learn3(NET_PARAM, const NeuronalNet::MultiSignalVector* expectedOutputVec);
UNITY_NET_API void BackpropNet_learn4(NET_PARAM, const NeuronalNet::SignalVector* expectedOutputVec);
UNITY_NET_API void BackpropNet_learn5(NET_PARAM, size_t streamIndex, const NeuronalNet::SignalVector* expectedOutputVec);

UNITY_NET_API const NeuronalNet::SignalVector* BackpropNet_getError1(NET_PARAM, size_t streamIndex);
UNITY_NET_API const NeuronalNet::MultiSignalVector* BackpropNet_getError2(NET_PARAM, const NeuronalNet::MultiSignalVector* expectedOutputVec);
UNITY_NET_API const NeuronalNet::SignalVector* BackpropNet_getError3(NET_PARAM, size_t streamIndex, const NeuronalNet::SignalVector* expectedOutputVec);
UNITY_NET_API const NeuronalNet::MultiSignalVector* BackpropNet_getError4(NET_PARAM);

#endif