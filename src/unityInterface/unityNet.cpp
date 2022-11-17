#include "unityInterface/unityNet.h"

#ifdef MAKE_DLL
#define NULL_PTR_RETURN(ptr, returnVal) \
	if(!ptr) return returnVal;
#define NULL_PTR_RETURN_VOID(ptr) \
	if(!ptr) return;

using namespace NeuronalNet;
void* Net_instantiate()
{
	return new Net();
}
void Net_dealocate(NET_PARAM)
{
	delete net;
}

size_t Net_getVersion_major()
{
	return Net::getVersion_major();
}
size_t Net_getVersion_minor()
{
	return Net::getVersion_minor();
}
size_t Net_getVersion_patch()
{
	return Net::getVersion_patch();
}

void Net_setDimensions(NET_PARAM, size_t inputs, size_t hiddenX, size_t hiddenY, size_t outputs)
{
	NULL_PTR_RETURN_VOID(net);
	net->setDimensions(inputs, hiddenX, hiddenY, outputs);
}
void Net_setStreamSize(NET_PARAM, size_t size)
{
	NULL_PTR_RETURN_VOID(net);
	net->setStreamSize(size);
}
size_t Net_getStreamSize(NET_PARAM)
{
	NULL_PTR_RETURN(net, 0);
	return net->getStreamSize();
}
size_t Net_getInputCount(NET_PARAM)
{
	NULL_PTR_RETURN(net, 0);
	return net->getInputCount();
}
size_t Net_getHiddenXCount(NET_PARAM)
{
	NULL_PTR_RETURN(net, 0);
	return net->getHiddenXCount();
}
size_t Net_getHiddenYCount(NET_PARAM)
{
	NULL_PTR_RETURN(net, 0);
	return net->getHiddenYCount();
}
size_t Net_getOutputCount(NET_PARAM)
{
	NULL_PTR_RETURN(net, 0);
	return net->getOutputCount();
}
size_t Net_getNeuronCount(NET_PARAM)
{
	NULL_PTR_RETURN(net, 0);
	return net->getNeuronCount();
}

void Net_setActivation(NET_PARAM, NeuronalNet::Activation act)
{
	NULL_PTR_RETURN_VOID(net);
	net->setActivation(act);
}
Activation Net_getActivation(NET_PARAM)
{
	NULL_PTR_RETURN(net, Activation::linear);
	return net->getActivation();
}
void Net_setHardware(NET_PARAM, NeuronalNet::Hardware ware)
{
	NULL_PTR_RETURN_VOID(net);
	net->setHardware(ware);
}
Hardware Net_getHardware(NET_PARAM)
{
	NULL_PTR_RETURN(net, Hardware::cpu);
	return net->getHardware();
}
void Net_enableBias(NET_PARAM, bool enable)
{
	NULL_PTR_RETURN_VOID(net);
	net->enableBias(enable);
}
bool Net_isBiasEnabled(NET_PARAM)
{
	NULL_PTR_RETURN(net, false);
	return net->isBiasEnabled();
}

bool Net_build(NET_PARAM)
{
	NULL_PTR_RETURN(net, false);
	return net->build();
}
bool Net_isBuilt(NET_PARAM)
{
	NULL_PTR_RETURN(net, false);
	return net->isBuilt();
}
void Net_randomizeWeights1(NET_PARAM)
{
	NULL_PTR_RETURN_VOID(net);
	net->randomizeWeights();
}
bool Net_randomizeWeights2(NET_PARAM, size_t from, size_t to)
{
	NULL_PTR_RETURN(net, false);
	return net->randomizeWeights(from, to);
}
float Net_getRandomValue(float min, float max)
{
	return Net::getRandomValue(min, max);
}
void Net_randomizeBias(NET_PARAM)
{
	NULL_PTR_RETURN_VOID(net);
	net->randomizeBias();
}
void Net_randomize(float* list, size_t size, float min, float max)
{
	Net::randomize(list, size, min, max);
}



void Net_setInputVector1(NET_PARAM, float* signalList)
{
	NULL_PTR_RETURN_VOID(net);
	net->setInputVector(signalList);
}
void Net_setInputVector2(NET_PARAM, size_t stream, float* signalList)
{
	NULL_PTR_RETURN_VOID(net);
	net->setInputVector(stream, signalList);
}
void Net_setInputVector3(NET_PARAM, const SignalVector* signalList)
{
	NULL_PTR_RETURN_VOID(net);
	net->setInputVector(*signalList);
}
void Net_setInputVector4(NET_PARAM, size_t stream, const SignalVector* signalList)
{
	NULL_PTR_RETURN_VOID(net);
	net->setInputVector(stream, *signalList);
}
void Net_setInputVector5(NET_PARAM, const MultiSignalVector* streamVector)
{
	NULL_PTR_RETURN_VOID(net);
	net->setInputVector(*streamVector);
}



void Net_setInput1(NET_PARAM, size_t input, float signal)
{
	NULL_PTR_RETURN_VOID(net);
	net->setInput(input, signal);
}
void Net_setInput2(NET_PARAM, size_t stream, size_t input, float signal)
{
	NULL_PTR_RETURN_VOID(net);
	net->setInput(stream, input, signal);
}
float Net_getInput1(NET_PARAM, size_t input)
{
	NULL_PTR_RETURN(net, 0);
	return net->getInput(input);
}
float Net_getInput2(NET_PARAM, size_t stream, size_t input)
{
	NULL_PTR_RETURN(net, 0);
	return net->getInput(stream, input);
}
const SignalVector* Net_getInputVector(NET_PARAM, size_t stream)
{
	NULL_PTR_RETURN(net, nullptr);
	return &net->getInputVector(stream);
}
const MultiSignalVector* Net_getInputStreamVector(NET_PARAM)
{
	NULL_PTR_RETURN(net, nullptr);
	return &net->getInputStreamVector();
}
const SignalVector* Net_getOutputVector(NET_PARAM, size_t stream)
{
	NULL_PTR_RETURN(net, nullptr);
	return &net->getOutputVector(stream);
}
const MultiSignalVector* Net_getOutputStreamVector(NET_PARAM)
{
	NULL_PTR_RETURN(net, nullptr);
	return &net->getOutputStreamVector();
}
float Net_GetOutput1(NET_PARAM, size_t output)
{
	NULL_PTR_RETURN(net, 0);
	return net->getOutput(output);
}
float Net_GetOutput2(NET_PARAM, size_t stream, size_t output)
{
	NULL_PTR_RETURN(net, 0);
	return net->getOutput(stream, output);
}

// MultiSignalVector Net_getNetinputStreamVector(NET_PARAM);
// MultiSignalVector Net_getNeuronValueStreamVector(NET_PARAM);

void Net_setWeight1(NET_PARAM, size_t layer, size_t neuron, size_t input, float weight)
{
	NULL_PTR_RETURN_VOID(net);
	net->setWeight(layer, neuron, input, weight);
}
void Net_setWeight2(NET_PARAM, const std::vector<float>* list)
{
	NULL_PTR_RETURN_VOID(net);
	net->setWeight(*list);
}
void Net_setWeight3(NET_PARAM, const float* list)
{
	NULL_PTR_RETURN_VOID(net);
	net->setWeight(list);
}
void Net_setWeight4(NET_PARAM, const float* list, size_t to)
{
	NULL_PTR_RETURN_VOID(net);
	net->setWeight(list, to);
}
void Net_setWeight5(NET_PARAM, const float* list, size_t insertOffset, size_t count)
{
	NULL_PTR_RETURN_VOID(net);
	net->setWeight(list, insertOffset, count);
}
float Net_getWeight1(NET_PARAM, size_t layer, size_t neuron, size_t input)
{
	NULL_PTR_RETURN(net, 0);
	return net->getWeight(layer, neuron, input);
}
const float* Net_getWeight2(NET_PARAM)
{
	NULL_PTR_RETURN(net, nullptr);
	return net->getWeight();
}
size_t Net_getWeightSize(NET_PARAM)
{
	NULL_PTR_RETURN(net, 0);
	net->getWeightSize();
}
void Net_setBias1(NET_PARAM, size_t layer, size_t neuron, float bias)
{
	NULL_PTR_RETURN_VOID(net);
	net->setBias(layer, neuron, bias);
}
void Net_setBias2(NET_PARAM, const std::vector<float>* list)
{
	NULL_PTR_RETURN_VOID(net);
	net->setBias(*list);
}
void Net_setBias3(NET_PARAM, const float* list)
{
	NULL_PTR_RETURN_VOID(net);
	net->setBias(list);
}
float Net_getBias1(NET_PARAM, size_t layer, size_t neuron)
{
	NULL_PTR_RETURN(net, 0);
	return net->getBias(layer, neuron);
}
const float* Net_getBias2(NET_PARAM)
{
	NULL_PTR_RETURN(net, nullptr);
	net->getBias();
}


void Net_calculate1(NET_PARAM)
{
	NULL_PTR_RETURN_VOID(net);
	net->calculate();
}
void Net_calculate2(NET_PARAM, size_t stream)
{
	NULL_PTR_RETURN_VOID(net);
	net->calculate(stream);
}
void Net_calculate3(NET_PARAM, size_t streamBegin, size_t streamEnd)
{
	NULL_PTR_RETURN_VOID(net);
	net->calculate(streamBegin, streamEnd);
}
#endif