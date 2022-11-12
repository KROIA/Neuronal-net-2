#pragma once
#include "unityConfig.h"
#include "backend/net.h"

#ifdef MAKE_DLL
#define NET_PARAM NeuronalNet::Net* net

UNITY_NET_API void* Net_instantiate();
UNITY_NET_API void Net_dealocate(NET_PARAM);

UNITY_NET_API size_t Net_getVersion_major();
UNITY_NET_API size_t Net_getVersion_minor();
UNITY_NET_API size_t Net_getVersion_patch();

UNITY_NET_API void Net_setDimensions(NET_PARAM, size_t inputs, size_t hiddenX, size_t hiddenY, size_t outputs);
UNITY_NET_API void Net_setStreamSize(NET_PARAM, size_t size);
UNITY_NET_API size_t Net_getStreamSize(NET_PARAM);
UNITY_NET_API size_t Net_getInputCount(NET_PARAM);
UNITY_NET_API size_t Net_getHiddenXCount(NET_PARAM);
UNITY_NET_API size_t Net_getHiddenYCount(NET_PARAM);
UNITY_NET_API size_t Net_getOutputCount(NET_PARAM);
UNITY_NET_API size_t Net_getNeuronCount(NET_PARAM);

UNITY_NET_API void Net_setActivation(NET_PARAM, NeuronalNet::Activation act);
UNITY_NET_API NeuronalNet::Activation Net_getActivation(NET_PARAM);

UNITY_NET_API void Net_setHardware(NET_PARAM, NeuronalNet::Hardware ware);
UNITY_NET_API NeuronalNet::Hardware Net_getHardware(NET_PARAM);

UNITY_NET_API void Net_enableBias(NET_PARAM, bool enable);
UNITY_NET_API bool Net_isBiasEnabled(NET_PARAM);

UNITY_NET_API bool Net_build(NET_PARAM);
UNITY_NET_API bool Net_isBuilt(NET_PARAM);
UNITY_NET_API void Net_randomizeWeights1(NET_PARAM);
UNITY_NET_API bool Net_randomizeWeights2(NET_PARAM, size_t from, size_t to);
UNITY_NET_API float Net_getRandomValue(float min, float max);
UNITY_NET_API void Net_randomizeBias(NET_PARAM);
UNITY_NET_API void Net_randomize(float* list, size_t size, float min, float max);

UNITY_NET_API void Net_setInputVector1(NET_PARAM, float* signalList);
UNITY_NET_API void Net_setInputVector2(NET_PARAM, size_t stream, float* signalList);
UNITY_NET_API void Net_setInputVector3(NET_PARAM, const NeuronalNet::SignalVector* signalList);
UNITY_NET_API void Net_setInputVector4(NET_PARAM, size_t stream, const NeuronalNet::SignalVector* signalList);
UNITY_NET_API void Net_setInputVector5(NET_PARAM, const NeuronalNet::MultiSignalVector* streamVector);

UNITY_NET_API void Net_setInput1(NET_PARAM, size_t input, float signal);
UNITY_NET_API void Net_setInput2(NET_PARAM, size_t stream, size_t input, float signal);
UNITY_NET_API float Net_getInput1(NET_PARAM, size_t input);
UNITY_NET_API float Net_getInput2(NET_PARAM, size_t stream, size_t input);
UNITY_NET_API const NeuronalNet::SignalVector* Net_getInputVector(NET_PARAM, size_t stream = 0);
UNITY_NET_API const NeuronalNet::MultiSignalVector* Net_getInputStreamVector(NET_PARAM);
UNITY_NET_API const NeuronalNet::SignalVector* Net_getOutputVector(NET_PARAM, size_t stream = 0);
UNITY_NET_API const NeuronalNet::MultiSignalVector* Net_getOutputStreamVector(NET_PARAM);
UNITY_NET_API float Net_GetOutput1(NET_PARAM, size_t output);
UNITY_NET_API float Net_GetOutput2(NET_PARAM, size_t stream, size_t output);


// UNITY_NET_API NeuronalNet::MultiSignalVector Net_getNetinputStreamVector(NET_PARAM);
// UNITY_NET_API NeuronalNet::MultiSignalVector Net_getNeuronValueStreamVector(NET_PARAM);


UNITY_NET_API void Net_setWeight1(NET_PARAM, size_t layer, size_t neuron, size_t input, float weight);
UNITY_NET_API void Net_setWeight2(NET_PARAM, const std::vector<float>* list);
UNITY_NET_API void Net_setWeight3(NET_PARAM, const float* list);
UNITY_NET_API void Net_setWeight4(NET_PARAM, const float* list, size_t to);
UNITY_NET_API void Net_setWeight5(NET_PARAM, const float* list, size_t insertOffset, size_t count);
UNITY_NET_API float Net_getWeight1(NET_PARAM, size_t layer, size_t neuron, size_t input);
UNITY_NET_API const float* Net_getWeight2(NET_PARAM);
UNITY_NET_API size_t Net_getWeightSize(NET_PARAM);
UNITY_NET_API void Net_setBias1(NET_PARAM, size_t layer, size_t neuron, float bias);
UNITY_NET_API void Net_setBias2(NET_PARAM, float* list);
UNITY_NET_API float Net_getBias1(NET_PARAM, size_t layer, size_t neuron);
UNITY_NET_API const float* Net_getBias2(NET_PARAM);


UNITY_NET_API void Net_calculate1(NET_PARAM);
UNITY_NET_API void Net_calculate2(NET_PARAM, size_t stream);
UNITY_NET_API void Net_calculate3(NET_PARAM, size_t streamBegin, size_t streamEnd);
//UNITY_NET_API void Net_graphics_update(NET_PARAM, const vector<GraphicsNeuronInterface*>& graphicsNeuronInterfaceList,
//					 const vector<GraphicsConnectionInterface*>& graphicsConnectionInterfaceList,
//					 size_t streamIndex);

#endif
