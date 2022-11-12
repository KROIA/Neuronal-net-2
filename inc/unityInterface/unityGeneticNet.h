#pragma once
#include "unityConfig.h"
#include "backend/geneticNet.h"

#ifdef MAKE_DLL
#define NET_PARAM NeuronalNet::GeneticNet* net

UNITY_NET_API void* GeneticNet_instantiate(size_t netCount);
UNITY_NET_API void GeneticNet_dealocate(NET_PARAM);

UNITY_NET_API size_t GeneticNet_getNetCount(NET_PARAM);

UNITY_NET_API void GeneticNet_setDimensions(NET_PARAM, size_t inputs, size_t hiddenX, size_t hiddenY, size_t outputs);
UNITY_NET_API void GeneticNet_setStreamSize(NET_PARAM, size_t size);
UNITY_NET_API size_t GeneticNet_getStreamSize(NET_PARAM);
UNITY_NET_API size_t GeneticNet_getInputCount(NET_PARAM);
UNITY_NET_API size_t GeneticNet_getHiddenXCount(NET_PARAM);
UNITY_NET_API size_t GeneticNet_getHiddenYCount(NET_PARAM);
UNITY_NET_API size_t GeneticNet_getOutputCount(NET_PARAM);
UNITY_NET_API size_t GeneticNet_getNeuronCount(NET_PARAM);

UNITY_NET_API void GeneticNet_setActivation(NET_PARAM, NeuronalNet::Activation act);
UNITY_NET_API NeuronalNet::Activation GeneticNet_getActivation(NET_PARAM);

UNITY_NET_API void GeneticNet_setHardware(NET_PARAM, enum NeuronalNet::Hardware ware);
UNITY_NET_API NeuronalNet::Hardware GeneticNet_getHardware(NET_PARAM);

UNITY_NET_API void GeneticNet_enableBias(NET_PARAM, bool enable);
UNITY_NET_API bool GeneticNet_isBiasEnabled(NET_PARAM);

UNITY_NET_API bool GeneticNet_build(NET_PARAM);
UNITY_NET_API bool GeneticNet_isBuilt(NET_PARAM);
UNITY_NET_API void GeneticNet_randomizeWeights1(NET_PARAM);
UNITY_NET_API bool GeneticNet_randomizeWeights2(NET_PARAM, size_t from, size_t to);
UNITY_NET_API void GeneticNet_randomizeBias(NET_PARAM);

UNITY_NET_API void GeneticNet_setInputVector1(NET_PARAM, size_t netIndex, float* signalList);
UNITY_NET_API void GeneticNet_setInputVector2(NET_PARAM, size_t netIndex, size_t stream, float* signalList);
UNITY_NET_API void GeneticNet_setInputVector3(NET_PARAM, size_t netIndex, const NeuronalNet::SignalVector* signalList);
UNITY_NET_API void GeneticNet_setInputVector4(NET_PARAM, size_t netIndex, size_t stream, const NeuronalNet::SignalVector* signalList);
UNITY_NET_API void GeneticNet_setInputVector5(NET_PARAM, size_t netIndex, const NeuronalNet::MultiSignalVector* streamVector);

UNITY_NET_API void GeneticNet_setInput1(NET_PARAM, size_t netIndex, size_t input, float signal);
UNITY_NET_API void GeneticNet_setInput2(NET_PARAM, size_t netIndex, size_t stream, size_t input, float signal);
UNITY_NET_API float GeneticNet_getInput1(NET_PARAM, size_t netIndex, size_t input);
UNITY_NET_API float GeneticNet_getInput2(NET_PARAM, size_t netIndex, size_t stream, size_t input);
UNITY_NET_API const NeuronalNet::SignalVector* GeneticNet_getInputVector(NET_PARAM, size_t netIndex, size_t stream = 0);
UNITY_NET_API const NeuronalNet::MultiSignalVector* GeneticNet_getInputStreamVector(NET_PARAM, size_t netIndex);
UNITY_NET_API const NeuronalNet::SignalVector* GeneticNet_getOutputVector(NET_PARAM, size_t netIndex, size_t stream = 0);
UNITY_NET_API const NeuronalNet::MultiSignalVector* GeneticNet_getOutputStreamVector(NET_PARAM, size_t netIndex);
UNITY_NET_API float GeneticNet_GetOutput1(NET_PARAM, size_t netIndex, size_t output);
UNITY_NET_API float GeneticNet_GetOutput2(NET_PARAM, size_t netIndex, size_t stream, size_t output);

//MultiSignalVector getNetinputStreamVector(size_t netIndex);
//MultiSignalVector getNeuronValueStreamVector(size_t netIndex);

UNITY_NET_API void GeneticNet_setWeight1(NET_PARAM, size_t netIndex, size_t layer, size_t neuron, size_t input, float weight);
UNITY_NET_API void GeneticNet_setWeight2(NET_PARAM, size_t netIndex, const std::vector<float>* list);
UNITY_NET_API void GeneticNet_setWeight3(NET_PARAM, size_t netIndex, const float* list);
UNITY_NET_API void GeneticNet_setWeight4(NET_PARAM, size_t netIndex, const float* list, size_t to);
UNITY_NET_API void GeneticNet_setWeight5(NET_PARAM, size_t netIndex, const float* list, size_t insertOffset, size_t count);
UNITY_NET_API float GeneticNet_getWeight1(NET_PARAM, size_t netIndex, size_t layer, size_t neuron, size_t input);
UNITY_NET_API const float* GeneticNet_getWeight2(NET_PARAM, size_t netIndex);
UNITY_NET_API size_t GeneticNet_getWeightSize(NET_PARAM);
UNITY_NET_API void GeneticNet_setBias1(NET_PARAM, size_t netIndex, size_t layer, size_t neuron, float bias);
UNITY_NET_API void GeneticNet_setBias2(NET_PARAM, size_t netIndex, float* list);
UNITY_NET_API float GeneticNet_getBias1(NET_PARAM, size_t netIndex, size_t layer, size_t neuron);
UNITY_NET_API const float* GeneticNet_getBias2(NET_PARAM, size_t netIndex);



UNITY_NET_API NeuronalNet::Net* GeneticNet_getNet(NET_PARAM, size_t index);
UNITY_NET_API void GeneticNet_setMutationChance(NET_PARAM, float chance); // 0 no chance, 1 every time
UNITY_NET_API float GeneticNet_getMutationChance(NET_PARAM);
UNITY_NET_API void GeneticNet_setMutationFactor(NET_PARAM, float radius); // a +- value for the min max range of random mutation. w = deltaW + oldW
UNITY_NET_API float GeneticNet_getMutationFactor(NET_PARAM);
UNITY_NET_API void GeneticNet_calculate(NET_PARAM);
UNITY_NET_API void GeneticNet_learn1(NET_PARAM, const std::vector<float>* ranks); // Ranks must be positive otherwise they will be set to 0
UNITY_NET_API void GeneticNet_learn2(NET_PARAM, const float* ranks); // Ranks must be positive otherwise they will be set to 0


#endif