#include "unityInterface/unityNetSerializer.h"

#ifdef MAKE_DLL
#define NULL_PTR_RETURN(ptr, returnVal) \
	if(!ptr) return returnVal;
#define NULL_PTR_RETURN_VOID(ptr) \
	if(!ptr) return;

using namespace NeuronalNet;

void* NetSerializer_instantiate()
{
	return new NetSerializer();
}
void NetSerializer_dealocate(SERIALIZER_PARAM)
{
	delete serializer;
}

void NetSerializer_setFilePath(SERIALIZER_PARAM, const char* path)
{
	NULL_PTR_RETURN_VOID(serializer);
	serializer->setFilePath(path);
}
const char* NetSerializer_getFilePath(SERIALIZER_PARAM)
{
	NULL_PTR_RETURN(serializer, nullptr);
	return serializer->getFilePath().c_str();
}

bool NetSerializer_saveToFile1(SERIALIZER_PARAM, NeuronalNet::Net* net)
{
	NULL_PTR_RETURN(serializer, false);
	return serializer->saveToFile(net);
}
bool NetSerializer_saveToFile2(SERIALIZER_PARAM, NeuronalNet::GeneticNet* net)
{
	NULL_PTR_RETURN(serializer, false);
	return serializer->saveToFile(net);
}

bool NetSerializer_readFromFile1(SERIALIZER_PARAM, NeuronalNet::Net* net)
{
	NULL_PTR_RETURN(serializer, false);
	return serializer->readFromFile(net);
}
bool NetSerializer_readFromFile2(SERIALIZER_PARAM, NeuronalNet::GeneticNet* net)
{
	NULL_PTR_RETURN(serializer, false);
	return serializer->readFromFile(net);
}
#endif