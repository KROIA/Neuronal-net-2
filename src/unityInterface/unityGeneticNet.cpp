#include "unityInterface/unityGeneticNet.h"

#ifdef MAKE_DLL
#define NULL_PTR_RETURN(ptr, returnVal) \
	if(!ptr) return returnVal;
#define NULL_PTR_RETURN_VOID(ptr) \
	if(!ptr) return;

using namespace NeuronalNet;

void* GeneticNet_instantiate(size_t netCount)
{
	return new GeneticNet(netCount);
}
void GeneticNet_dealocate(NET_PARAM)
{
	delete net;
}


void GeneticNet_setNetCount(NET_PARAM, size_t netCount)
{
	NULL_PTR_RETURN_VOID(net);
	net->setNetCount(netCount);
}
size_t GeneticNet_getNetCount(NET_PARAM)
{
	NULL_PTR_RETURN(net, 0);
	return net->getNetCount();
}

void GeneticNet_setDimensions(NET_PARAM, size_t inputs, size_t hiddenX, size_t hiddenY, size_t outputs)
{
	NULL_PTR_RETURN_VOID(net);
	net->setDimensions(inputs, hiddenX, hiddenY, outputs);
}
void GeneticNet_setStreamSize(NET_PARAM, size_t size)
{
	NULL_PTR_RETURN_VOID(net);
	net->setStreamSize(size);
}
size_t GeneticNet_getStreamSize(NET_PARAM)
{
	NULL_PTR_RETURN(net, 0);
	return net->getStreamSize();
}
size_t GeneticNet_getInputCount(NET_PARAM)
{
	NULL_PTR_RETURN(net, 0);
	return net->getInputCount();
}
size_t GeneticNet_getHiddenXCount(NET_PARAM)
{
	NULL_PTR_RETURN(net, 0);
	return net->getHiddenXCount();
}
size_t GeneticNet_getHiddenYCount(NET_PARAM)
{
	NULL_PTR_RETURN(net, 0);
	return net->getHiddenYCount();
}
size_t GeneticNet_getOutputCount(NET_PARAM)
{
	NULL_PTR_RETURN(net, 0);
	return net->getOutputCount();
}
size_t GeneticNet_getNeuronCount(NET_PARAM)
{
	NULL_PTR_RETURN(net, 0);
	return net->getNeuronCount();
}


void GeneticNet_setActivation(NET_PARAM, NeuronalNet::Activation act)
{
	NULL_PTR_RETURN_VOID(net);
	net->setActivation(act);
}
Activation GeneticNet_getActivation(NET_PARAM)
{
	NULL_PTR_RETURN(net, Activation::linear);
	return net->getActivation();
}
void GeneticNet_setHardware(NET_PARAM, NeuronalNet::Hardware ware)
{
	NULL_PTR_RETURN_VOID(net);
	net->setHardware(ware);
}
Hardware GeneticNet_getHardware(NET_PARAM)
{
	NULL_PTR_RETURN(net, Hardware::cpu);
	return net->getHardware();
}
void GeneticNet_enableBias(NET_PARAM, bool enable)
{
	NULL_PTR_RETURN_VOID(net);
	net->enableBias(enable);
}
bool GeneticNet_isBiasEnabled(NET_PARAM)
{
	NULL_PTR_RETURN(net, false);
	return net->isBiasEnabled();
}

void GeneticNet_unbuild(NET_PARAM)
{
	NULL_PTR_RETURN_VOID(net);
	return net->unbuild();
}
bool GeneticNet_build(NET_PARAM)
{
	NULL_PTR_RETURN(net, false);
	return net->build();
}
bool GeneticNet_isBuilt(NET_PARAM)
{
	NULL_PTR_RETURN(net, false);
	return net->isBuilt();
}
void GeneticNet_randomizeWeights1(NET_PARAM)
{
	NULL_PTR_RETURN_VOID(net);
	net->randomizeWeights();
}
bool GeneticNet_randomizeWeights2(NET_PARAM, size_t from, size_t to)
{
	NULL_PTR_RETURN(net, false);
	return net->randomizeWeights(from, to);
}
void GeneticNet_randomizeBias(NET_PARAM)
{
	NULL_PTR_RETURN_VOID(net);
	net->randomizeBias();
}

void GeneticNet_setInputVector1(NET_PARAM, size_t netIndex, float* signalList)
{
	NULL_PTR_RETURN_VOID(net);
	net->setInputVector(netIndex, signalList);
}
void GeneticNet_setInputVector2(NET_PARAM, size_t netIndex, size_t stream, float* signalList)
{
	NULL_PTR_RETURN_VOID(net);
	net->setInputVector(stream, signalList);
}
void GeneticNet_setInputVector3(NET_PARAM, size_t netIndex, const SignalVector* signalList)
{
	NULL_PTR_RETURN_VOID(net);
	net->setInputVector(netIndex, *signalList);
}
void GeneticNet_setInputVector4(NET_PARAM, size_t netIndex, size_t stream, const SignalVector* signalList)
{
	NULL_PTR_RETURN_VOID(net);
	net->setInputVector(netIndex, stream, *signalList);
}
void GeneticNet_setInputVector5(NET_PARAM, size_t netIndex, const MultiSignalVector* streamVector)
{
	NULL_PTR_RETURN_VOID(net);
	net->setInputVector(netIndex, *streamVector);
}
void GeneticNet_setInput1(NET_PARAM, size_t netIndex, size_t input, float signal)
{
	NULL_PTR_RETURN_VOID(net);
	net->setInput(netIndex, input, signal);
}
void GeneticNet_setInput2(NET_PARAM, size_t netIndex, size_t stream, size_t input, float signal)
{
	NULL_PTR_RETURN_VOID(net);
	net->setInput(netIndex, stream, input, signal);
}
float GeneticNet_getInput1(NET_PARAM, size_t netIndex, size_t input)
{
	NULL_PTR_RETURN(net, 0);
	return net->getInput(netIndex, input);
}
float GeneticNet_getInput2(NET_PARAM, size_t netIndex, size_t stream, size_t input)
{
	NULL_PTR_RETURN(net, 0);
	return net->getInput(netIndex, stream, input);
}
const SignalVector* GeneticNet_getInputVector(NET_PARAM, size_t netIndex, size_t stream)
{
	NULL_PTR_RETURN(net, nullptr);
	return &net->getInputVector(netIndex, stream);
}
const MultiSignalVector* GeneticNet_getInputStreamVector(NET_PARAM, size_t netIndex)
{
	NULL_PTR_RETURN(net, nullptr);
	return &net->getInputStreamVector(netIndex);
}
const SignalVector* GeneticNet_getOutputVector(NET_PARAM, size_t netIndex, size_t stream)
{
	NULL_PTR_RETURN(net, nullptr);
	return &net->getOutputVector(netIndex, stream);
}
const MultiSignalVector* GeneticNet_getOutputStreamVector(NET_PARAM, size_t netIndex)
{
	NULL_PTR_RETURN(net, nullptr);
	return &net->getOutputStreamVector(netIndex);
}
float GeneticNet_GetOutput1(NET_PARAM, size_t netIndex, size_t output)
{
	NULL_PTR_RETURN(net, 0);
	return net->getOutput(netIndex, output);
}
float GeneticNet_GetOutput2(NET_PARAM, size_t netIndex, size_t stream, size_t output)
{
	NULL_PTR_RETURN(net, 0);
	return net->getOutput(netIndex, stream, output);
}

// MultiSignalVector GeneticNet_getNetinputStreamVector(NET_PARAM);
// MultiSignalVector GeneticNet_getNeuronValueStreamVector(NET_PARAM);

void GeneticNet_setWeight1(NET_PARAM, size_t netIndex, size_t layer, size_t neuron, size_t input, float weight)
{
	NULL_PTR_RETURN_VOID(net);
	net->setWeight(netIndex, layer, neuron, input, weight);
}
void GeneticNet_setWeight2(NET_PARAM, size_t netIndex, const std::vector<float>* list)
{
	NULL_PTR_RETURN_VOID(net);
	net->setWeight(netIndex, *list);
}
void GeneticNet_setWeight3(NET_PARAM, size_t netIndex, const float* list)
{
	NULL_PTR_RETURN_VOID(net);
	net->setWeight(netIndex, list);
}
void GeneticNet_setWeight4(NET_PARAM, size_t netIndex, const float* list, size_t to)
{
	NULL_PTR_RETURN_VOID(net);
	net->setWeight(netIndex, list, to);
}
void GeneticNet_setWeight5(NET_PARAM, size_t netIndex, const float* list, size_t insertOffset, size_t count)
{
	NULL_PTR_RETURN_VOID(net);
	net->setWeight(netIndex, list, insertOffset, count);
}
float GeneticNet_getWeight1(NET_PARAM, size_t netIndex, size_t layer, size_t neuron, size_t input)
{
	NULL_PTR_RETURN(net, 0);
	return net->getWeight(netIndex, layer, neuron, input);
}
const float* GeneticNet_getWeight2(NET_PARAM, size_t netIndex)
{
	NULL_PTR_RETURN(net, nullptr);
	return net->getWeight(netIndex);
}
size_t GeneticNet_getWeightSize(NET_PARAM)
{
	NULL_PTR_RETURN(net, 0);
	net->getWeightSize();
}
void GeneticNet_setBias1(NET_PARAM, size_t netIndex, size_t layer, size_t neuron, float bias)
{
	NULL_PTR_RETURN_VOID(net);
	net->setBias(netIndex, layer, neuron, bias);
}
void GeneticNet_setBias2(NET_PARAM, size_t netIndex, const std::vector<float>* list)
{
	NULL_PTR_RETURN_VOID(net);
	net->setBias(netIndex, *list);
}
void GeneticNet_setBias3(NET_PARAM, size_t netIndex, float* list)
{
	NULL_PTR_RETURN_VOID(net);
	net->setBias(netIndex, list);
}
float GeneticNet_getBias1(NET_PARAM, size_t netIndex, size_t layer, size_t neuron)
{
	NULL_PTR_RETURN(net, 0);
	return net->getBias(netIndex, layer, neuron);
}
const float* GeneticNet_getBias2(NET_PARAM, size_t netIndex)
{
	NULL_PTR_RETURN(net, nullptr);
	net->getBias(netIndex);
}



Net* GeneticNet_getNet(NET_PARAM, size_t index)
{
	NULL_PTR_RETURN(net, nullptr);
	return net->getNet(index);
}
void GeneticNet_setMutationChance(NET_PARAM, float chance)
{
	NULL_PTR_RETURN_VOID(net);
	net->setMutationChance(chance);
}
float GeneticNet_getMutationChance(NET_PARAM)
{
	NULL_PTR_RETURN(net, 0);
	return net->getMutationFactor();
}
void GeneticNet_setMutationFactor(NET_PARAM, float radius)
{
	NULL_PTR_RETURN_VOID(net);
	net->setMutationFactor(radius);
}
float GeneticNet_getMutationFactor(NET_PARAM)
{
	NULL_PTR_RETURN(net, 0);
	return net->getMutationFactor();
}
void GeneticNet_setWeightBounds(NET_PARAM, float radius)
{
	NULL_PTR_RETURN_VOID(net);
	net->setWeightBounds(radius);
}
float GeneticNet_getWeightBounds(NET_PARAM)
{
	NULL_PTR_RETURN(net, 0);
	return net->getWeightBounds();
}
void GeneticNet_calculate(NET_PARAM)
{
	NULL_PTR_RETURN_VOID(net);
	net->calculate();
}
void GeneticNet_learn1(NET_PARAM, const std::vector<float>*ranks)
{
	NULL_PTR_RETURN_VOID(net);
	net->learn(*ranks);
}
void GeneticNet_learn2(NET_PARAM, const float* ranks)
{
	NULL_PTR_RETURN_VOID(net);
	std::vector<float> r(net->getNetCount());
	for (size_t i = 0; i < r.size(); ++i)
	{
		r[i] = ranks[i];
	}
	net->learn(r);
}



#endif