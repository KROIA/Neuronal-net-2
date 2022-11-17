#include "unityInterface/unityBackpropNet.h"

#ifdef MAKE_DLL
#define NULL_PTR_RETURN(ptr, returnVal) \
	if(!ptr) return returnVal;
#define NULL_PTR_RETURN_VOID(ptr) \
	if(!ptr) return;

using namespace NeuronalNet;
void* BackpropNet_instantiate()
{
	return new BackpropNet();
}
void BackpropNet_dealocate(NET_PARAM)
{
	delete net;
}

void BackpropNet_unbuild(NET_PARAM)
{
	NULL_PTR_RETURN_VOID(net);
	return net->unbuild();
}
bool BackpropNet_build(NET_PARAM)
{
	NULL_PTR_RETURN(net, false);
	return net->build();
}

void BackpropNet_setLearnParameter(NET_PARAM, float learnParam)
{
	NULL_PTR_RETURN_VOID(net);
	net->setLearnParameter(learnParam);
}
float BackpropNet_getLearnParameter(NET_PARAM)
{
	NULL_PTR_RETURN(net, 0);
	return net->getLearnParameter();
}

void BackpropNet_setExpectedOutput1(NET_PARAM, const MultiSignalVector *expectedOutputVec)
{
	NULL_PTR_RETURN_VOID(net);
	net->setExpectedOutput(*expectedOutputVec);
}
void BackpropNet_setExpectedOutput2(NET_PARAM, const SignalVector *expectedOutputVec)
{
	NULL_PTR_RETURN_VOID(net);
	net->setExpectedOutput(*expectedOutputVec);
}
void BackpropNet_setExpectedOutput3(NET_PARAM, size_t streamIndex, const SignalVector * expectedOutputVec)
{
	NULL_PTR_RETURN_VOID(net);
	net->setExpectedOutput(streamIndex, *expectedOutputVec);
}
void BackpropNet_learn1(NET_PARAM)
{
	NULL_PTR_RETURN_VOID(net);
	net->learn();
}
void BackpropNet_learn2(NET_PARAM, size_t streamIndex)
{
	NULL_PTR_RETURN_VOID(net);
	net->learn(streamIndex);
}
void BackpropNet_learn3(NET_PARAM, const MultiSignalVector * expectedOutputVec)
{
	NULL_PTR_RETURN_VOID(net);
	net->learn(*expectedOutputVec);
}
void BackpropNet_learn4(NET_PARAM, const SignalVector * expectedOutputVec)
{
	NULL_PTR_RETURN_VOID(net);
	net->learn(*expectedOutputVec);
}
void BackpropNet_learn5(NET_PARAM, size_t streamIndex, const SignalVector * expectedOutputVec)
{
	NULL_PTR_RETURN_VOID(net);
	net->learn(streamIndex, *expectedOutputVec);
}

const SignalVector* BackpropNet_getError1(NET_PARAM, size_t streamIndex)
{
	NULL_PTR_RETURN(net, nullptr);
	return &net->getError(streamIndex);
}
const MultiSignalVector* BackpropNet_getError2(NET_PARAM, const MultiSignalVector * expectedOutputVec)
{
	NULL_PTR_RETURN(net, nullptr);
	return &net->getError(*expectedOutputVec);
}
const SignalVector* BackpropNet_getError3(NET_PARAM, size_t streamIndex, const SignalVector * expectedOutputVec)
{
	NULL_PTR_RETURN(net, nullptr);
	return &net->getError(streamIndex, *expectedOutputVec);
}
const MultiSignalVector* BackpropNet_getError4(NET_PARAM)
{
	NULL_PTR_RETURN(net, nullptr);
	return &net->getError();
}

#endif